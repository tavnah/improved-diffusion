
import os
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path, PosixPath
from skimage import color
from skimage.transform import resize
import cv2
import torch


def find_similar_patch_from_one_image(patch, image):

    result = cv2.matchTemplate(image, patch, cv2.TM_CCOEFF)
    #result = cv2.matchTemplate(image, patch, cv2.TM_CCORR)
    #result = cv2.matchTemplate(image, patch, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    match_top_left = max_loc
    # Get the size of the template
    template_height, template_width = patch.shape
    # Get the coordinates of the bottom-right corner of the matched patch
    match_bottom_right = (match_top_left[0] + template_width, match_top_left[1] + template_height)
    # Extract the matched patch from the original image
    matched_patch = image[match_top_left[1]:match_bottom_right[1], match_top_left[0]:match_bottom_right[0]]
    return matched_patch, max_val

def find_similar_patch_augment(patch, image):
    patch0 = patch
    tens_patch = torch.tensor(patch)
    patch1 = (tens_patch.fliplr()).numpy()
    patch2 = cv2.rotate(patch, cv2.ROTATE_180)
    patch3 = cv2.rotate(patch1, cv2.ROTATE_180)

    patches = [patch0, patch1, patch2, patch3]
    current_max = -1000000000
    for patch in patches:
        similar_patch, max_val = find_similar_patch_from_one_image(patch, image)
        if current_max < max_val:
            current_max = max_val
            patch_couple = (patch, similar_patch, max_val)
    return patch_couple

def find_most_similar_patch(patch, images_folder_path, resize_factor=1):
    '''
    patch - black and white patch, uint8
    '''
    images_folder = PosixPath(images_folder_path)
    current_max = -1000
    for image in images_folder.iterdir():
        if image.is_file():
            image_path = image.__fspath__()
            image = plt.imread(image_path)
            if image.shape[-1] == 4:
                bw_image = color.rgb2gray(color.rgba2rgb(image))
            else:
                bw_image = color.rgb2gray(image)
            new_size = tuple([x // resize_factor for x in bw_image.shape])
            res_image = resize(bw_image, new_size)
            bw_image = res_image
            if np.max(bw_image) < 2:
                bw_image = np.uint8(bw_image * 255)

            (patch, similar_patch, max_val) = find_similar_patch_augment(patch, bw_image)
            if max_val > current_max:
                current_max = max_val
                patch_couple = (patch, similar_patch, max_val)

    return patch_couple

def find_similar_patch_for_generated_patches(generated_patches_path, image_folder_path, resize_factor=1):
    with np.load(generated_patches_path) as data:
        lst = data.files
        total_patches_num = len(data[lst[0]])
        rows_num = 3
        cols_num = total_patches_num
        plt.figure(figsize=(30, 10))
        for i, patch in enumerate(data[lst[0]]):
            bw_patch = color.rgb2gray(patch)
            if np.max(bw_patch) < 2:
                bw_patch = bw_patch * 255
                bw_patch = np.uint8(bw_patch)
            aug_patch, similar_patch, _ = find_most_similar_patch(bw_patch, image_folder_path, resize_factor)
            diff_patches = np.zeros((aug_patch.shape[0],aug_patch.shape[1],3))
            diff_patches[:,:,0] = aug_patch
            diff_patches[:,:,1] = similar_patch
            diff_patches = diff_patches.astype(np.uint8)

            plt.subplot(rows_num, cols_num, i+1)
            plt.imshow(aug_patch, cmap='gray')
            plt.title(f'patch {i}')
            plt.subplot(rows_num, cols_num, i+1+cols_num)
            plt.imshow(similar_patch, cmap='gray')
            plt.title(f'matched patch {i}')
            plt.subplot(rows_num, cols_num, i + 1 + 2*cols_num)
            plt.imshow(diff_patches)
            plt.title(f'diff {i}')

        plt.show()

def show_10_patches(generated_patches_path):
    with np.load(generated_patches_path) as data:
        lst = data.files
        total_patches_num = len(data[lst[0]])
        rows_num = 2
        cols_num = 5
        plt.figure(figsize=(30, 10))
        for i, patch in enumerate(data[lst[0]]):
            plt.subplot(rows_num, cols_num, i+1)
            plt.imshow(patch)
            plt.title(f'patch {i}')
        plt.show()

def check_script_with_exist_path(patch_path, images_folder_path, resize_factor=1):

    patch = plt.imread(patch_path)
    patch = color.rgb2gray(patch)
    if np.max(patch) < 2:
        patch = patch * 255
    patch = np.uint8(patch)
    patch_couple = find_most_similar_patch(patch, images_folder_path, resize_factor)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(patch_couple[0], cmap='gray')
    plt.title('patch')
    plt.subplot(1, 2, 2)
    plt.imshow(patch_couple[1], cmap='gray')
    plt.title('matched patch')
    plt.show()


if __name__ == "__main__":
    #patch_path = '/data/GAN_project/microtubules/shareloc/alpha_tubulin_scale_4/one_image/patches_o0.25/patch218_3.jpg'
    #images_folder_path = '/data/GAN_project/microtubules/shareloc/alpha_tubulin_scale_4/one_image/HR_image'
    #check_script_with_exist_path(patch_path,images_folder_path)


    orig_patches_folder = '/data/GAN_project/microtubules/onit/HR'
    #orig_patches_folder = '/data/GAN_project/mitochondria/onit/HR'
    orig_patches_folder = '/data/GAN_project/tiff_files/good'
    #orig_patches_folder = '/data/GAN_project/microtubules/shareloc/alpha_tubulin_scale_4/patches_ol0.25'
    #patch = plt.imread('/data/GAN_project/microtubules/onit/HR/try/patches/patch59.jpg')
    #image = plt.imread('/data/GAN_project/microtubules/onit/HR/try/microtubules_i_50_exp_t_30msec002 - STORM image.tif')
    #find_similar_patch(patch, orig_patches_folder)
    #orig_patches_folder = r'C:\Users\tav33\Downloads\shareloc'


    #patches_path = '/data/GAN_project/diffusion_tries/openai-2023-03-31-15-33-00-056364/samples_10x64x64x3.npz' #microtubules
    #patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-04-11-15-37-42-886600/samples_10x64x64x3.npz' #mitochondria
    patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-04-26-15-04-29-012512/samples_10x256x256x3.npz' #mitochondria 256 80000
    patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-04-29-17-40-37-521308/samples_10x256x256x3.npz' #mitochondria 256 10000
    patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-04-29-18-23-24-267052/samples_10x256x256x3.npz' #mitochondria 256 70000
    patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-04-30-08-06-41-095522/samples_10x256x256x3.npz' #mitochondria 256 80000+36000

    patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-05-01-21-53-15-163588/samples_10x256x256x3.npz' #microtubuls 256 v2 data
    patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-05-02-08-03-36-590047/samples_10x256x256x3.npz' # microtubuls 256 v2 data after lr 1e-6

    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-14-10-21-29-732845/samples_10x256x256x3.npz' #shareloc images
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-16-08-18-48-268574/samples_10x256x256x3.npz' #shareloc images, scale 4, longer run
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-18-23-30-07-355647/samples_10x256x256x3.npz' #shareloc images, scale 4, 2000 steps
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-19-08-22-07-172546/samples_10x256x256x3.npz' #shareloc images, scale 4, 2000 steps, breakpoint from one above
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-19-08-48-46-030054/samples_10x256x256x3.npz' # minimum loss of the above
    #patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-19-17-06-44-051174/samples_10x256x256x3.npz' #scale 4, 3000 steps
    #patches_path = r'C:\Users\tav33\Courses\ProjectGAN\data\patches_q0.01q0.99'
    #show_10_patches(patches_path)
    find_similar_patch_for_generated_patches(patches_path, orig_patches_folder,3 )

    # with np.load('/data/GAN_project/diffusion_tries/openai-2023-03-31-15-33-00-056364/samples_10x64x64x3.npz') as data:
    #    lst = data.files
    #    for patch in data[lst[0]]:
    #        find_similar_patch(patch, orig_patches_folder)
