# A collection of useful functions
from matplotlib.backends.backend_pdf import PdfPages

from scipy import sparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2
import torch
import random

import os
from pathlib import Path

from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import transform


#import gs
from PIL import Image

from sklearn.metrics import mean_squared_error
from scipy.spatial import distance

import subprocess

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import math


def define_reproducibility(seed=42):

    # Define the seed for operations on CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # GPU operations are deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy seed
    np.random.seed(seed)

    # Vanilla Python operation
    random.seed(seed)

    print(f"Reproducibility is set to {seed}.")

def load_pots(path, img_size = 256):
    '''
    A simple function to load the pots from a .npz file
    '''
    pots = sparse.load_npz(path).toarray().reshape(-1, img_size, img_size)
    return pots


def fit_and_transform_encoders(df, columns):
    '''
    Fit and transform the encoders for the specified columns    
    '''
    # Create an empty dictionary to store the encoders
    encoders = {}
    
    # Iterate over the columns
    for col in columns:
        # Create a LabelEncoder object and fit it to the column
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(df[col])
        
        # Create a OneHotEncoder object and fit it to the column
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(label_encoded.reshape(-1, 1))
        
        # Save the encoders to the dictionary
        encoders[col] = (label_encoder, one_hot_encoder)
    
    return encoders

def transform_data_using_encoders(data, encoders):
    '''
    Transform the data using the encoders
    '''
    transformed_columns = []
    
    for col, (label_encoder, one_hot_encoder) in encoders.items():
        # Apply Label Encoding
        label_encoded = label_encoder.transform(data[col])
        
        # Apply One-Hot Encoding
        one_hot_encoded = one_hot_encoder.transform(label_encoded.reshape(-1, 1))
        
        transformed_columns.append(pd.DataFrame(one_hot_encoded, columns=[f"{col}_encoded_{i}" for i in range(one_hot_encoded.shape[1])]))
    
    return pd.concat(transformed_columns, axis=1)


def minimum_image(img, margin=1):
    """
    Crop the minimum bounding box around a non-zero region in an image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        margin (int): Margin to add around the minimum bounding box.

    Returns:
        numpy.ndarray: Minimum image.

    """
    c_hull = np.where(img > 0)
    x_min, x_max = c_hull[0].min(), c_hull[0].max()
    y_min, y_max = c_hull[1].min(), c_hull[1].max()
    min_img = img[x_min:x_max,y_min:y_max]
    min_img = np.pad(min_img, margin)

    return min_img

def pad_image_fixed_dim_padded(img, dim):
    """
    Pad an image to a fixed dimension by adding padding around the image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        dim (int): Desired dimension for the padded image (height = width).

    Returns:
        numpy.ndarray: Padded image.

    """
    img_shape = img.shape
    dh = dim - img_shape[0]
    dw = dim - img_shape[1]

    if not dh % 2:
        a = (dh//2, dh//2)
    else:
        a = (dh//2, dh//2 + 1)

    if not dw % 2:
        b = (dw//2, dw//2)
    else:
        b = (dw//2, dw//2 + 1)

    padded = np.pad(img, (a, b))
    return padded

def minimum_image_bb(img):
    """
    Crop the minimum bounding box around a non-zero region in an image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        margin (int): Margin to add around the minimum bounding box.

    Returns:
        numpy.ndarray: Minimum image.

    """
    c_hull = np.where(img > 0)
    x_min, x_max = c_hull[0].min(), c_hull[0].max()
    y_min, x_max = c_hull[1].min(), c_hull[1].max()

    return x_min, y_min #MODIFICATO QUA
    #return x_max, y_min


def create_bounding_box(image, top_left_x, top_left_y, scale):
    '''
    Create a bounding box around the non-zero pixels in the image and resize it to 128x128 pixels.
    
    '''
    # Find the non-zero pixels in the image
    non_zero_pixels = np.argwhere(image > 0)
    
    if len(non_zero_pixels) == 0:
        raise ValueError("No non-zero pixels found in the image.")
    
    # Define the bounding box
    min_y, min_x = non_zero_pixels.min(axis=0)
    max_y, max_x = non_zero_pixels.max(axis=0)

    # Define the width and height of the bounding box
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Define the new width and height of the bounding box according to the specified scale
    new_width = int(width * scale)
    new_height = int(height * scale)

    new_top_left_x = int(top_left_x)
    new_top_left_y = int(top_left_y)

    # Create a new array of zeros of 128x128 pixels
    new_array = np.zeros((128, 128), dtype=np.uint8)

    # Be sure that the new bounding box is inside the new array
    if new_top_left_x < 0:
        new_top_left_x = 0
    if new_top_left_y < 0:
        new_top_left_y = 0
    if new_top_left_x + new_width > 128:
        new_width = 128 - new_top_left_x
    if new_top_left_y + new_height > 128:
        new_height = 128 - new_top_left_y

    # Extract the bounding box from the image
    bounding_box = image[min_y:max_y+1, min_x:max_x+1]
    
    # Resize the bounding box to the new dimensions
    resized_bounding_box = cv2.resize(bounding_box, (new_width, new_height))

    # Copy the resized bounding box into the new array
    new_array[new_top_left_y:new_top_left_y+new_height, new_top_left_x:new_top_left_x+new_width] = (
        resized_bounding_box * 255
    ).astype(np.uint8)

    return new_array


def kl_divergence(z_log_var, z_mean):
    return -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1) 


def read_img_files(base_dir, img_extensions=['.jpg', '.jpeg', '.png']):
    """
    Read image files from a directory.

    Parameters:
        base_dir (str or pathlib.Path): Base directory path.
        img_extensions (list): List of image file extensions to consider. Default is ['.jpg', '.jpeg', '.png'].

    Returns:
        list: List of image file paths.
    """
    base_dir = Path(base_dir)
    img_files = []
    for elem in os.listdir(base_dir):
        if os.path.isdir(base_dir /elem):
            for x in os.listdir(base_dir / elem):
                if (base_dir / elem / x).suffix in img_extensions:
                    img_files.append(base_dir / elem / x)
        elif  (base_dir / elem).suffix in img_extensions:
            img_files.append(base_dir / elem)
        else:
            print(f"something strange happened for file {elem}")
    img_files.sort()
    return img_files

def denoise_and_binarize(img):
    """
    Denoise an image using morphological operations and then binarize it using Otsu's thresholding.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.

    Returns:
        numpy.ndarray: Binarized image.

    """
    selem = morphology.disk(3)
    x = img > threshold_otsu(img)
    x = morphology.closing(x, selem)

    return x



def reconstruction_metrics(original, reconstructed):
    
    metrics_list = []

    pot_numpy_flatten = np.vstack([np.array(data).ravel() for data in original])
    decoded_numpy_flatten = np.vstack([np.array(data).ravel() for data in reconstructed])

    rlts_real = gs.rlts(pot_numpy_flatten, gamma=1.0/128, n=pot_numpy_flatten.shape[0])

    rlts_fake = gs.rlts(decoded_numpy_flatten, gamma=1.0/128, n=decoded_numpy_flatten.shape[0])

    os.makedirs("original", exist_ok=True)
    os.makedirs("reconstructed", exist_ok=True)

    for i, (origin, rec) in enumerate(zip(original, reconstructed)):
        origin_normalized = origin.astype(np.uint8) * 255
        rec_normalized = rec.astype(np.uint8) * 255

        original_path = os.path.join("original", f"original_{i}.png")
        Image.fromarray(origin_normalized).save(original_path)

        reconstructed_path = os.path.join("reconstructed", f"reconstructed_{i}.png")
        Image.fromarray(rec_normalized).save(reconstructed_path)

        metrics_list.append([np.sqrt(mean_squared_error(origin.flatten(), rec.flatten())), 
                            1 - distance.dice(origin.flatten(), rec.flatten())])

    # Calcola il FID score utilizzando subprocess
    fid_process = subprocess.Popen(["python", "-m", "pytorch_fid", "--dim", "64", "--device", "cuda", "reconstructed/", "original/"], stdout=subprocess.PIPE)
    fid_output, _ = fid_process.communicate()

    # Elimina le immagini con os
    os.system("rm -r original")
    os.system("rm -r reconstructed")

    # Estrai il FID score dalla stringa di output
    fid_score = float(fid_output.decode().split()[-1])

    metrics_df = pd.DataFrame(metrics_list, columns=["RMSE", "Dice coefficient"]).mean()
    metrics_df["GS score"] = gs.geom_score(rlts_fake, rlts_real)
    metrics_df["FID score"] = fid_score

    metrics_df = metrics_df.to_frame()

    metrics_df.columns = ["Value"]

    return metrics_df




def plot_batches(batches_list, title_list=None, figsize=(12, 8), nrow=8, title_size=12, tick_size=12, save_fig=False):

    fig, ax = plt.subplots(1, len(batches_list), figsize=figsize, sharex=False, sharey=False)
    for i, batch in enumerate(batches_list):
        batch_im = make_grid(batch.float(), nrow=nrow, padding=0)
        
        ax[i].imshow(batch_im.permute(1,2,0).cpu().numpy())
        ax[i].set_aspect('equal', adjustable='box')


        for n in range(len(ax)):
            y_positions = np.arange(math.ceil(len(batch) / nrow)) * 128 + 64

            ax[n].set_yticks(y_positions)

            try:
                if len(y_positions) % 2 == 0:
                    ax[n].set_yticklabels(np.arange(1, math.ceil(len(batch) / nrow) + 2), fontsize=tick_size)
                else:
                    ax[n].set_yticklabels(np.arange(1, math.ceil(len(batch) / nrow) + 1), fontsize=tick_size)
            except ValueError as e:
                    ax[n].set_yticklabels(np.arange(1, math.ceil(len(batch) / nrow +1)), fontsize=tick_size)

            x_positions = np.arange(nrow) * 128 + 64  

            # Set the x-tick labels and positions
            ax[n].set_xticks(x_positions)
            letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
            ax[n].set_xticklabels(letters[:nrow], fontsize=tick_size)

            # Set the title for each subplot
            if title_list is not None:
                ax[n].set_title(title_list[n], fontsize=title_size)

    if save_fig:
        
        plt.savefig("batches.png", dpi=300, bbox_inches='tight')




                

def create_pdf_first_eval(original_images, pdf_filename, id_df, images_per_page=5):
    a4_width_inches = 8.27
    a4_height_inches = 11.69

    num_columns = 2
    
    n_images = len(original_images)
    num_pages = int(np.ceil(n_images / images_per_page/2))

    with PdfPages(pdf_filename) as pdf:
        for page in range(num_pages):
            fig, ax = plt.subplots(images_per_page, num_columns, figsize=(a4_width_inches, a4_height_inches))

            for i in range(images_per_page*2):
                index = page * images_per_page*2 + i
                if index < n_images and index < len(original_images):
                    ax.flatten()[i].imshow(original_images.img[index].reshape(128,128), cmap='gray')
                    ax.flatten()[i].set_title(f"ID: {id_df[index]}")
                    ax.flatten()[i].axis('off')
            

            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def create_pdf_second_eval(original_images, reconstructed_images, pdf_filename, id_df, images_per_page=5):
    a4_width_inches = 8.27
    a4_height_inches = 11.69

    num_columns = 2
    
    n_images = len(original_images)
    num_pages = int(np.ceil(n_images / images_per_page))

    with PdfPages(pdf_filename) as pdf:
        for page in range(num_pages):
            fig, ax = plt.subplots(images_per_page, num_columns, figsize=(a4_width_inches, a4_height_inches))

            for i in range(images_per_page):
                index = page * images_per_page + i
                if index < n_images:
                    ax[i, 0].imshow(original_images[index], cmap='gray')
                    ax[i, 0].set_title(f"ID: {id_df[index]}")
                    ax[i, 1].imshow(reconstructed_images[index], cmap='gray')
                    ax[i, 1].set_title(f"ID: {id_df[index]}")
                    ax[i, 0].axis('off')
                    ax[i, 1].axis('off')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def pad_image_postprocessing(img, dim):
    """
    Pad an image to a fixed dimension by adding padding around the image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        dim (int): Desired dimension for the padded image (height = width).
        margin (int): Size of the margin to add around the image.

    Returns:
        numpy.ndarray: Padded image.

    """
    img_shape = img.shape
    dh = max(dim - img_shape[0], 0)
    dw = max(dim - img_shape[1], 0)

    # Calcola il padding uniforme su tutti i lati dell'immagine
    top_pad = max((dh) // 2, 0)
    bottom_pad = max(dh, 0)
    left_pad = max((dw) // 2, 0)
    right_pad = max(dw - left_pad, 0)

    # Aggiungi il padding uniforme su tutti i lati dell'immagine
    padded = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)

    return padded


def predict_fragments(dataloader, model, batch_size, df_ids, context_name = "OSTS", device="cuda", ):

    os.makedirs(f"img_pos_{context_name}", exist_ok=True)

    for batch, (pot, _, _, _, _, archeo_info, _, _) in enumerate(dataloader):

        pot = pot.to(device).float()
        
        if len(archeo_info) != 0:
            archeo_info = archeo_info.to(device).float()

        if len(archeo_info) != 0:
            archeo_info = archeo_info.view(batch_size, archeo_info.shape[1], 1, 1)
            concat_images = torch.cat((pot, archeo_info.expand(-1, -1, 128, 128)), dim=1)
            predicted_dimensions = model(concat_images)
        else:
            predicted_dimensions = model(pot)

        predicted_dimensions_array = predicted_dimensions.cpu().detach().numpy().squeeze()


        for i in range(len(pot)):
            arr_new = create_bounding_box(pot[i].cpu().numpy().squeeze(), predicted_dimensions_array[i][1]*128, predicted_dimensions_array[i][0]*128, predicted_dimensions_array[i][2]*2.5)
            im = Image.fromarray(255-arr_new)
            im.save(f"img_pos_{context_name}/{df_ids.ids.values[i+ (batch*batch_size)]}.png") 


def read_and_denoise_imgs(path):
    files_path = read_img_files(path)

    imgs_red = []
    for img_file in files_path:
        x = Image.open(img_file).convert("L")
        x = np.array(x)
        x = 255 - x
        x = denoise_and_binarize(x)
        imgs_red.append(x)

    imgs_red = np.array(imgs_red)
    return imgs_red