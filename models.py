### Autoencoders are based on https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/1_VAE_mnist_sigmoid_mse.ipynb 


import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import rescale

from torch import nn

from utils import *

from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights

import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score

class CustomModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomModel, self).__init__()
        self.base_resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        if num_classes is not None:
            self.base_resnet.conv1 = nn.Conv2d(1 + num_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.base_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_resnet.fc = nn.Linear(2048, 3)


    def forward(self, x):

        x = self.base_resnet(x)
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.GELU(), #ReLU
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 4 * 4 * 256),
            nn.GELU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)), 
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0),  
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        decoded = self.decoder(x)
        return decoded


class CustomLossMSE(nn.Module):
    def __init__(self, mask_weight):
        super(CustomLossMSE, self).__init__()
        self.mask_weight = mask_weight

    def forward(self, original, reconstructed, mask):
        # Define the loss for the masked part
        masked_diff = (original * mask - reconstructed * mask) ** 2
        masked_loss = torch.sum(masked_diff)
        
        # Define the loss for the unmasked part
        unmasked_diff = (original * (1 - mask) - reconstructed * (1 - mask)) ** 2
        unmasked_loss = torch.sum(unmasked_diff)
        
        # Define the total loss as a weighted sum
        total_loss = self.mask_weight * masked_loss + (1 - self.mask_weight) * unmasked_loss
        
        return total_loss
    
class CustomLossBCE(nn.Module):
    def __init__(self, mask_weight):
        super(CustomLossBCE, self).__init__()
        self.mask_weight = mask_weight

    def forward(self, original, reconstructed, mask):
        # Define the loss for the masked part using BCE
        masked_bce_loss = - (original * mask * torch.log(reconstructed * mask + 1e-9) + (1 - original) * mask * torch.log(1 - reconstructed * mask + 1e-9))
        masked_loss = torch.sum(masked_bce_loss)
        
        # Define the loss for the unmasked part with BCE
        unmasked_bce_loss = - (original * (1 - mask) * torch.log(reconstructed * (1 - mask) + 1e-9) + (1 - original) * (1 - mask) * torch.log(1 - reconstructed * (1 - mask) + 1e-9))
        unmasked_loss = torch.sum(unmasked_bce_loss)
        
        # Define the total loss as a weighted sum
        total_loss = self.mask_weight * masked_loss + (1 - self.mask_weight) * unmasked_loss
        
        return total_loss
    

class RegressorFineTuner():
    def __init__(self, model, epochs, data_loader_train, data_loader_test, lr=0.0005, BATCH_SIZE=128, 
                 device='cuda', SCALE_CONSTANT=30, POS_Y_CONSTANT=1, POS_X_CONSTANT=1):

        self.model = model
        self.epochs = epochs
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.BATCH_SIZE = BATCH_SIZE

        self.device = device
        self.SCALE_CONSTANT = SCALE_CONSTANT
        self.POS_Y_CONSTANT = POS_Y_CONSTANT
        self.POS_X_CONSTANT = POS_X_CONSTANT

        self.train_losses = []
        self.test_losses = []


        self.pretrained_model_criterion = nn.MSELoss()
        self.pretrained_model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            ### Append only the last epoch predictions and real total dimensions
            self.predictions = []
            self.real_total_dimensions = []
            ### Set the model to train mode
            self.model.train()
            
            ### Set the training loss to 0
            total_loss_training = 0

            for batch, (pot, masked_img, padded_tensor, bounding_box, scale_factor, archeo_info, _, _) in enumerate(self.data_loader_train):
                ### Move the data to the device (GPU if available) and modify the bounding box coordinates and scale factor according to the constants.
                padded_tensor = padded_tensor.to(self.device).float()
                bounding_box = bounding_box.to(self.device).float()
                if len(archeo_info) != 0:
                    archeo_info = archeo_info.to(self.device).float()

                scale_factor = scale_factor.to(self.device).float()/2.5 # * self.SCALE_CONSTANT
                bounding_box[:, 0] = bounding_box[:, 0]/128# * self.POS_Y_CONSTANT
                bounding_box[:, 1] = bounding_box[:, 1]/128# * self.POS_X_CONSTANT

                total_dimensions = torch.cat((bounding_box, scale_factor.unsqueeze(1)), dim=1)
                
                if len(archeo_info) != 0:
                ### Concatenate the padded tensor with the archeological information
                    archeo_info = archeo_info.view(BATCH_SIZE, archeo_info.shape[1], 1, 1)
                    concat_images = torch.cat((padded_tensor, archeo_info.expand(-1, -1, 128, 128)), dim=1)
                    ### Forward pass with the concatenated images
                    predicted_dimensions = self.model(concat_images)
                else:
                    ### Forward pass with the padded tensor
                    predicted_dimensions = self.model(padded_tensor)


                ### Compute the loss
                loss = self.pretrained_model_criterion(predicted_dimensions, total_dimensions) 

                ### Backward pass
                self.pretrained_model_optimizer.zero_grad()
                loss.backward()
                self.pretrained_model_optimizer.step()
                
                ### Update the training loss
                total_loss_training += loss.item()
            
            ### Define the average training loss and append it to the list
            average_loss = total_loss_training / len(data_loader_train)
            self.train_losses.append(average_loss)

            ### Set the model to evaluation mode
            self.model.eval()
            ### Set the test loss to 0
            total_test_loss = 0
            with torch.no_grad():
                for batch, (pot, masked_img, padded_tensor, bounding_box, scale_factor, archeo_info, _, _) in enumerate(self.data_loader_test):
                    
                    ### Move the data to the device (GPU if available) and modify the bounding box coordinates and scale factor according to the constants.
                    padded_tensor = padded_tensor.to(self.device).float()
                    bounding_box = bounding_box.to(self.device).float()
                    if len(archeo_info) != 0:
                        archeo_info = archeo_info.to(self.device).float()

                    scale_factor = scale_factor.to(self.device).float()/2.5# * self.SCALE_CONSTANT
                    bounding_box[:, 0] = bounding_box[:, 0]/128# * self.POS_Y_CONSTANT
                    bounding_box[:, 1] = bounding_box[:, 1]/128# * self.POS_X_CONSTANT

                    total_dimensions = torch.cat((bounding_box, scale_factor.unsqueeze(1)), dim=1)

                    
                    if len(archeo_info) != 0:
                    ### Concatenate the padded tensor with the archeological information
                        archeo_info = archeo_info.view(BATCH_SIZE, archeo_info.shape[1], 1, 1)
                        concat_images = torch.cat((padded_tensor, archeo_info.expand(-1, -1, 128, 128)), dim=1)
                        ### Forward pass with the concatenated images
                        predicted_dimensions = self.model(concat_images)
                    else:
                        ### Forward pass with the padded tensor
                        predicted_dimensions = self.model(padded_tensor)

                    ### Compute the loss
                    loss = self.pretrained_model_criterion(predicted_dimensions, total_dimensions)

                    ### Save test information and update the test loss
                    self.predictions.extend(predicted_dimensions.cpu().numpy())
                    self.real_total_dimensions.extend(total_dimensions.cpu().numpy())
                    total_test_loss += loss.item()
            ### Define the average test loss and append it to the list
            average_loss_test = total_test_loss / len(data_loader_test)
            self.test_losses.append(average_loss_test)

            ### Print the epoch, the training loss and the test loss 
            if epoch % 5 == 0:  
                tqdm.write(f'Epoch [{epoch + 1}/{self.epochs}] - Training Loss: {average_loss:.4f} - Testing Loss: {average_loss_test:.4f}')

    def plot_losses(self, figsize=(8, 4)):
        
        fig, ax = plt.subplots(figsize=figsize)

        plt.plot(np.array(self.train_losses), label='Training loss', marker='o', linestyle='-')
        plt.plot(np.array(self.test_losses), label='Test loss', marker='o', linestyle='-')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.legend(frameon=True, loc='upper right', fontsize='large')

        # Display a grid for readability
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def r_square_plot(self):

        # Set the figure size and create subplots with equal aspect ratio
        fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharex=False, sharey=False)
        for axis in ax:
            axis.grid(True)
            axis.set_aspect('equal', adjustable='box')

        ### convert to numpy array
        predictions_array = np.array(self.predictions)
        real_bb_array = np.array(self.real_total_dimensions)

        sns.scatterplot(x=real_bb_array[:, 0], y=predictions_array[:, 0], ax=ax[0])
        sns.scatterplot(x=real_bb_array[:, 1], y=predictions_array[:, 1], ax=ax[1])
        
        
        sns.scatterplot(x=real_bb_array[:, 2], y=predictions_array[:, 2], ax=ax[2], label='Values')
        ### add identity line
        x_0 = np.linspace(*ax[0].get_xlim())
        x_1 = np.linspace(*ax[1].get_xlim())
        x_2 = np.linspace(*ax[2].get_xlim())
        ax[0].plot(x_0, x_0, color='black', linestyle='--')
        ax[1].plot(x_1, x_1, color='black', linestyle='--')
        ax[2].plot(x_2, x_2, color='black', linestyle='--',  label='Identity Line')

        r_score_height = r2_score(real_bb_array[:, 0], predictions_array[:, 0])
        r_score_width = r2_score(real_bb_array[:, 1], predictions_array[:, 1])

        #r_score_scale = r2_score(real_bb_array[:, 2]/self.SCALE_CONSTANT, predictions_array[:, 2]/self.SCALE_CONSTANT)
        r_score_scale = r2_score(real_bb_array[:, 2], predictions_array[:, 2])

        ax[0].set_xlabel(f"Real Height \n $R^2$: {r_score_height:.2f}")
        ax[0].set_ylabel("Predicted Height")
        ax[1].set_xlabel(f"Real Width \n $R^2$: {r_score_width:.2f}")
        ax[1].set_ylabel("Predicted Width")
        ax[2].set_xlabel(f"Real Scale \n $R^2$: {r_score_scale:.2f}")
        ax[2].set_ylabel("Predicted Scale")

        ### add legend outside of plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);



class DAEFineTuner():
    def __init__(self, model, epochs, data_loader_train, data_loader_test, lr=0.0005, mask_weight=0.2, BATCH_SIZE=128, device='cuda'):
        ### Create a list to store the losses over training

        self.model = model
        self.epochs = epochs
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        
        self.loss = CustomLossMSE(mask_weight=mask_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)


        self.train_losses = []
        self.test_losses = []
        self.outputs_train = []
        self.outputs_test = []


        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            ### Set the model to train mode
            self.model.train()
            total_loss_train = 0
            for batch, (pot, masked, padded_tensor, bounding_box, scale_factor, archeo_info, mask, blurred_tensor) in enumerate(self.data_loader_train):
                
                ### Move the data to the device (GPU if available).
                pot = pot.to(device).float()
                blurred_tensor = blurred_tensor.to(self.device).float()
                

                ### Invert the mask
                inverted_mask = ~mask
                inverted_mask = inverted_mask.to(self.device).float()
                
                ### Forwards pass        
                decoded = self.model(blurred_tensor)                                
                        

                ### Calculate the pixelwise loss
                loss_total_train = self.loss(pot, decoded, inverted_mask)


                ### Backpropagation
                self.optimizer.zero_grad()
                loss_total_train.backward()
                self.optimizer.step()

                ### Update the total loss   
                total_loss_train += loss_total_train.item()

            ### Calculate the average loss over the epoch
            average_loss = total_loss_train / len(data_loader_train)
            
            ### Append the average loss to the list of losses
            self.train_losses.append(average_loss)
            self.outputs_train.append((epoch, decoded, blurred_tensor, pot))

            ### Set the model to evaluation mode
            self.model.eval()

            total_test_loss = 0


            with torch.no_grad():
                for batch_idx, (pot, masked, padded_tensor, bounding_box, scale_factor, archeo_info, mask, blurred_tensor) in enumerate(self.data_loader_test):
                    
                    ### Move the data to the device (GPU if available).
                    pot = pot.to(device).float()
                    blurred_tensor = blurred_tensor.to(self.device).float()
                    

                    ### Invert the mask
                    inverted_mask = ~mask
                    inverted_mask = inverted_mask.to(self.device).float()
                    
                    ### Forwards pass        
                    decoded = self.model(blurred_tensor)                                
                            

                    ### Calculate the pixelwise loss
                    loss_total_test = self.loss(pot, decoded, inverted_mask)

                    total_test_loss += loss_total_test.item()
                
                ### Calculate the average loss over the epoch
                average_loss_test = total_test_loss / len(data_loader_test)

                ### Append the average loss to the list of losses
                self.test_losses.append(average_loss_test)
                self.outputs_test.append((epoch, decoded, blurred_tensor, pot))

            if epoch % 5 == 0:  
               tqdm.write(f'Epoch [{epoch + 1}/{self.epochs}] - Training Loss: {average_loss:.4f} - Testing Loss: {average_loss_test:.4f}')

    def plot_losses(self, figsize=(8, 4)):
        
        fig, ax = plt.subplots(figsize=figsize)

        plt.plot(np.array(self.train_losses), label='Training loss', marker='o', linestyle='-')
        plt.plot(np.array(self.test_losses), label='Test loss', marker='o', linestyle='-')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.legend(frameon=True, loc='upper right', fontsize='large')

        # Display a grid for readability
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_outputs(self, epochs, output_set = "test", figsize=(15, 5)):

        for k in epochs:
            
            if output_set == "test":
                decoded_tensor = self.outputs_train[k][1].to("cpu").detach()
                blurred_tensor = self.outputs_train[k][2].to("cpu").detach()
                pot_tensor = self.outputs_train[k][3].to("cpu").detach()
            elif output_set == "train":
                decoded_tensor = self.outputs_test[k][1].to("cpu").detach()
                blurred_tensor = self.outputs_test[k][2].to("cpu").detach()
                pot_tensor = self.outputs_test[k][3].to("cpu").detach()
            else:
                raise ValueError("set must be either 'train' or 'test'")



            decoded = decoded_tensor.numpy()
            blurred = blurred_tensor.numpy()
            pot = pot_tensor.numpy()


            plt_number = 10

            fig, ax = plt.subplots(3, int(plt_number), figsize=figsize,
            subplot_kw={'xticks':[], 'yticks':[]},
            gridspec_kw=dict(hspace=0.5, wspace=0.1))
            
            if output_set == "train":
                plt.suptitle(f"Train epoch {k+1}")
            elif output_set == "test":
                plt.suptitle(f"Test epoch {k+1}")
            else:
                raise ValueError("output_set must be either 'train' or 'test'")

            for i in range(int(plt_number)):
                ax[0, i].imshow(np.transpose(pot[i], (1,2,0)), cmap='binary_r')
                ax[1, i].imshow(np.transpose(blurred[i],(1,2,0)), cmap='binary_r')
                ax[2, i].imshow(np.transpose(decoded[i],(1,2,0)), cmap='binary_r')
                ax[0, 0].set_ylabel('Original', fontsize=8)
                ax[1, 0].set_ylabel('Blurred', fontsize=8)
                ax[2, 0].set_ylabel('Reconstructed', fontsize=8)
                ax[2, i].set_xlabel(i)


class VAEFineTuner():
    def __init__(self, model, epochs, data_loader_train, data_loader_test, loss = "MSE", 
                 lr=0.0005, mask_weight=0.2, BATCH_SIZE=128, device='cuda'):
        
        self.model = model
        self.epochs = epochs
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.loss = loss
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        if self.loss == "MSE":
            self.loss = CustomLossMSE(mask_weight=mask_weight)
        elif self.loss == "BCE":
            self.loss = CustomLossBCE(mask_weight=mask_weight)
        else:
            raise ValueError("Loss must be either MSE or BCE")
                
        ### Create a list to store the losses over training
        self.train_losses = []
        self.test_losses = []
        self.outputs_train = []
        self.outputs_test = []


        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            ### Set the model to train mode
            self.model.train()
            total_loss_train = 0
            for batch, (pot, masked, padded_tensor, bounding_box, scale_factor, archeo_info, mask, _) in enumerate(self.data_loader_train):
                
                ### Move the data to the device (GPU if available).
                pot = pot.to(self.device).float()
                masked = masked.to(self.device).float()

                ### Invert the mask
                inverted_mask = ~mask
                inverted_mask = inverted_mask.to(self.device).float()
                
                ### Forwards pass        
                encoded, z_mean, z_log_var, decoded = self.model(masked)                                
                        
                ### Calculate the KL divergence
                kl_div = kl_divergence(z_log_var, z_mean)
                #batchsize = kl_div.size(0)
                kl_div = kl_div.mean() 

                ### Calculate the pixelwise loss
                pixelwise = self.loss(pot, decoded, inverted_mask)

                ### Calculate the total loss
                loss_total_train = pixelwise + kl_div


                ### Backpropagation
                self.optimizer.zero_grad()
                loss_total_train.backward()
                self.optimizer.step()

                ### Update the total loss   
                total_loss_train += loss_total_train.item()

            ### Calculate the average loss over the epoch
            average_loss = total_loss_train / len(data_loader_train)
            
            ### Append the average loss to the list of losses
            self.train_losses.append(average_loss)
            self.outputs_train.append((epoch, decoded, masked, pot))

            ### Set the model to evaluation mode
            self.model.eval()

            total_test_loss = 0


            with torch.no_grad():
                for batch_idx, (pot, masked, padded_tensor, bounding_box, scale_factor, archeo_info, mask, _) in enumerate(self.data_loader_test):
                    
                    ### Move the data to the device (GPU if available).
                    pot = pot.to(self.device).float()
                    masked = masked.to(self.device).float()
                    
                    ### Invert the mask
                    inverted_mask = ~mask
                    inverted_mask = inverted_mask.to(self.device).float()
                    
                    ### Forwards pass        
                    encoded, z_mean, z_log_var, decoded = self.model(masked)                                
                            
                    ### Calculate the KL divergence
                    kl_div = kl_divergence(z_log_var, z_mean)
                    #batchsize = kl_div.size(0)
                    kl_div = kl_div.mean() 

                    ### Calculate the pixelwise loss
                    pixelwise = self.loss(pot, decoded, inverted_mask)

                    ### Calculate the total loss
                    loss_total_test = pixelwise + kl_div

                    total_test_loss += loss_total_test.item()
                
                ### Calculate the average loss over the epoch
                average_loss_test = total_test_loss / len(data_loader_test)

                ### Append the average loss to the list of losses
                self.test_losses.append(average_loss_test)
                self.outputs_test.append((epoch, decoded, masked, pot))

            if epoch % 5 == 0:
                tqdm.write(f'Epoch [{epoch + 1}/{self.epochs}] - Training Loss: {average_loss:.4f} - Testing Loss: {average_loss_test:.4f}')

    def plot_losses(self, figsize=(8, 4)):
        
        fig, ax = plt.subplots(figsize=figsize)

        plt.plot(np.array(self.train_losses), label='Training loss', marker='o', linestyle='-')
        plt.plot(np.array(self.test_losses), label='Test loss', marker='o', linestyle='-')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.legend(frameon=True, loc='upper right', fontsize='large')

        # Display a grid for readability
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_outputs(self, epochs, output_set = "test", figsize=(15, 5)):
        for k in epochs:
            
            if output_set == "test":
                decoded_tensor = self.outputs_train[k][1].to("cpu").detach()
                masked_tensor = self.outputs_train[k][2].to("cpu").detach()
                pot_tensor = self.outputs_train[k][3].to("cpu").detach()
            elif output_set == "train":
                decoded_tensor = self.outputs_test[k][1].to("cpu").detach()
                masked_tensor = self.outputs_test[k][2].to("cpu").detach()
                pot_tensor = self.outputs_test[k][3].to("cpu").detach()
            else:
                raise ValueError("set must be either 'train' or 'test'")


            row_mask = torch.all(torch.eq(masked_tensor, 0), dim=3, keepdim=True)
            col_mask = torch.all(torch.eq(masked_tensor, 0), dim=2, keepdim=True)
            x_filled = torch.where(row_mask, decoded_tensor, pot_tensor)
            decoded_filled = torch.where(col_mask, decoded_tensor, x_filled)

            decoded = decoded_tensor.numpy()
            masked = masked_tensor.numpy()
            pot = pot_tensor.numpy()
            decoded_filled = decoded_filled.numpy()

            plt_number = 10

            fig, ax = plt.subplots(4, int(plt_number), figsize=(15, 5),
            subplot_kw={'xticks':[], 'yticks':[]},
            gridspec_kw=dict(hspace=0.5, wspace=0.1))

            if output_set == "train":
                plt.suptitle(f"Train epoch {k+1}")
            elif output_set == "test":
                plt.suptitle(f"Test epoch {k+1}")
            else:
                raise ValueError("output_set must be either 'train' or 'test'")

            for i in range(int(plt_number)):
                ax[0, i].imshow(np.transpose(pot[i], (1,2,0)), cmap='binary_r')
                ax[1, i].imshow(np.transpose(masked[i],(1,2,0)), cmap='binary_r')
                ax[2, i].imshow(np.transpose(decoded[i],(1,2,0)), cmap='binary_r')
                ax[3, i].imshow(np.transpose(decoded_filled[i],(1,2,0)), cmap='binary_r')
                ax[0, 0].set_ylabel('Original', fontsize=8)
                ax[1, 0].set_ylabel('Real \n fragmented', fontsize=8)
                ax[2, 0].set_ylabel('Reconstructed', fontsize=8)
                ax[3, 0].set_ylabel('Reconstructed \n filled with \n original', fontsize=8)
                ax[3, i].set_xlabel(i)


class DAEFineTunerOriginal():
    def __init__(self, model, epochs, data_loader_train, data_loader_test, lr=0.0005, mask_weight=0.2, BATCH_SIZE=128, device='cuda'):
        ### Create a list to store the losses over training

        self.model = model
        self.epochs = epochs
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        
        self.loss = CustomLossMSE(mask_weight=mask_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)


        self.train_losses = []
        self.test_losses = []
        self.outputs_train = []
        self.outputs_test = []


        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            ### Set the model to train mode
            self.model.train()
            total_loss_train = 0
            for batch, (pot, masked, padded_tensor, bounding_box, scale_factor, archeo_info, mask, blurred_tensor) in enumerate(self.data_loader_train):
                
                ### Move the data to the device (GPU if available).
                pot = pot.to(device).float()
                blurred_tensor = blurred_tensor.to(self.device).float()
                

                ### Invert the mask
                inverted_mask = ~mask
                inverted_mask = inverted_mask.to(self.device).float()
                
                ### Forwards pass        
                decoded = self.model(blurred_tensor)                                
                        

                ### Calculate the pixelwise loss
                loss_total_train = self.loss(pot, decoded, inverted_mask)


                ### Backpropagation
                self.optimizer.zero_grad()
                loss_total_train.backward()
                self.optimizer.step()

                ### Update the total loss   
                total_loss_train += loss_total_train.item()

            ### Calculate the average loss over the epoch
            average_loss = total_loss_train / len(data_loader_train)
            
            ### Append the average loss to the list of losses
            self.train_losses.append(average_loss)
            self.outputs_train.append((epoch, decoded, blurred_tensor, pot))

            ### Set the model to evaluation mode
            self.model.eval()

            total_test_loss = 0


            with torch.no_grad():
                for batch_idx, (pot, masked, padded_tensor, bounding_box, scale_factor, archeo_info, mask, blurred_tensor) in enumerate(self.data_loader_test):
                    
                    ### Move the data to the device (GPU if available).
                    pot = pot.to(device).float()
                    blurred_tensor = blurred_tensor.to(self.device).float()
                    

                    ### Invert the mask
                    inverted_mask = ~mask
                    inverted_mask = inverted_mask.to(self.device).float()
                    
                    ### Forwards pass        
                    decoded = self.model(blurred_tensor)                                
                            

                    ### Calculate the pixelwise loss
                    loss_total_test = self.loss(pot, decoded, inverted_mask)

                    total_test_loss += loss_total_test.item()
                
                ### Calculate the average loss over the epoch
                average_loss_test = total_test_loss / len(data_loader_test)

                ### Append the average loss to the list of losses
                self.test_losses.append(average_loss_test)
                self.outputs_test.append((epoch, decoded, blurred_tensor, pot))


            tqdm.write(f'Epoch [{epoch + 1}/{self.epochs}] - Training Loss: {average_loss:.4f} - Testing Loss: {average_loss_test:.4f}')

    def plot_losses(self, figsize=(8, 4)):
        
        fig, ax = plt.subplots(figsize=figsize)

        plt.plot(np.array(self.train_losses), label='Training loss', marker='o', linestyle='-')
        plt.plot(np.array(self.test_losses), label='Test loss', marker='o', linestyle='-')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.legend(frameon=True, loc='upper right', fontsize='large')

        # Display a grid for readability
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_outputs(self, epochs, output_set = "test", figsize=(15, 5)):

        for k in epochs:
            
            if output_set == "test":
                decoded_tensor = self.outputs_train[k][1].to("cpu").detach()
                blurred_tensor = self.outputs_train[k][2].to("cpu").detach()
                pot_tensor = self.outputs_train[k][3].to("cpu").detach()
            elif output_set == "train":
                decoded_tensor = self.outputs_test[k][1].to("cpu").detach()
                blurred_tensor = self.outputs_test[k][2].to("cpu").detach()
                pot_tensor = self.outputs_test[k][3].to("cpu").detach()
            else:
                raise ValueError("set must be either 'train' or 'test'")



            decoded = decoded_tensor.numpy()
            blurred = blurred_tensor.numpy()
            pot = pot_tensor.numpy()


            plt_number = 10

            fig, ax = plt.subplots(3, int(plt_number), figsize=figsize,
            subplot_kw={'xticks':[], 'yticks':[]},
            gridspec_kw=dict(hspace=0.5, wspace=0.1))
            
            if output_set == "train":
                plt.suptitle(f"Train epoch {k+1}")
            elif output_set == "test":
                plt.suptitle(f"Test epoch {k+1}")
            else:
                raise ValueError("output_set must be either 'train' or 'test'")

            for i in range(int(plt_number)):
                ax[0, i].imshow(np.transpose(pot[i], (1,2,0)), cmap='binary_r')
                ax[1, i].imshow(np.transpose(blurred[i],(1,2,0)), cmap='binary_r')
                ax[2, i].imshow(np.transpose(decoded[i],(1,2,0)), cmap='binary_r')
                ax[0, 0].set_ylabel('Original', fontsize=8)
                ax[1, 0].set_ylabel('Blurred', fontsize=8)
                ax[2, 0].set_ylabel('Reconstructed', fontsize=8)
                ax[2, i].set_xlabel(i)


class reconstructionVAE(nn.Module):
    def __init__(self, image_latent_dims):
        super().__init__()
        self.latent_dims = image_latent_dims

        self.encoder = nn.Sequential( ### Input shape: (1, 128, 128)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  ### Output shape: (16, 64, 64) 
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), ### Output shape: (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), ### Output shape: (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), ### Output shape: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), ### Output shape: (256, 4, 4) 
            nn.BatchNorm2d(256),
            nn.GELU(),
            ###
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), ### Output shape: (256, 4, 4) 
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), ### Output shape: (256, 4, 4) 
            nn.BatchNorm2d(256),
            nn.GELU(),
            ###
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(image_latent_dims, 256),
            nn.GELU(),
            nn.Linear(256, 4 * 4 * 256), 
            nn.GELU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),  ### Output shape: (256, 4, 4)
            ###
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0), ### Output shape: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0), ### Output shape: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.GELU(),
            ###
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),  ### Output shape: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0), ### Output shape: (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0), ### Output shape: (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0), ### Output shape: (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, output_padding=0), ### Output shape: (1, 128, 128)
            nn.Sigmoid()
        )

        self.mu = nn.Linear(256, image_latent_dims)
        self.sigma = nn.Linear(256, image_latent_dims)




    def reparameterize_function(self, mu, sigma):
        eps = torch.randn(mu.size(0), mu.size(1)).to(mu.device)
        z = mu + eps * torch.exp(sigma / 2.)
        return z

    def forward(self, image_input):

        image_encoded = self.encoder(image_input)

        z_mean, z_log_var = self.mu(image_encoded), self.sigma(image_encoded)

        encoded = self.reparameterize_function(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

    def sample(self, n_samples, device):
        z = torch.randn((n_samples, self.latent_dims)).to(device)
        return self.decoder(z)