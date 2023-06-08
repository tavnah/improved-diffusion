import torchvision
import torchvision.transforms as transforms
from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from nd2reader import ND2Reader
#import matplotlib

#matplotlib.use('TkAgg')

def augmentation(img,n_rotation):
    '''
    the function creates augmentation of the same image: flipped L-R, and rotations.
    for each orientation (original + flipped) the function rotate by n_rotation.
    for example: if n_rotation = 6, 6 image will be created: 0 deg, 60 deg, 120 deg, 180 deg, 240 deg, 300 deg.
    :param img: pytorch tensor
    :param n_rotation: int
    :return: img_augmentations: pytorch tensor of tensors, each tensor is a new image
    '''
    flipped = img.fliplr()
    orientations = [img, flipped]
    img_augmentations = torch.zeros((n_rotation*2, img.shape[1], img.shape[2]))
    angle = 360/n_rotation
    for i, cur_img in enumerate(orientations):
        for j in range(n_rotation):
            cur_angle = j*angle
            rotated_img = transforms.functional.rotate(cur_img, interpolation=transforms.InterpolationMode.BILINEAR, angle= cur_angle)
            img_augmentations[(i*n_rotation) +j, :,:] = rotated_img

    return img_augmentations

def crop_to_patches(image, top, left, patch_height, patch_width, overlap):
    # overlap in precentage, for exmaple: 0.8 / 0.5
    overlap_range_width = int(patch_width * overlap)
    overlap_range_height = int(patch_height * overlap)
    patches_list = []
    cur_left = left


    #range_x = ((image.shape[1] - (image.shape[1] % patch_height)) // patch_height) * (1 + overlap)
    #range_y = ((image.shape[2] - (image.shape[2] % patch_width)) // patch_width) * (1 + overlap)
    range_x = 1 + (image.shape[1] - patch_height - (((image.shape[1] - patch_height)) % (patch_height - overlap_range_height))) // (patch_height-overlap_range_height)
    range_y = 1 + (image.shape[2] - patch_width - (((image.shape[2] - patch_width)) % (patch_width - overlap_range_width))) // (patch_width-overlap_range_width)

    for i in range(int(range_x)):
        for j in range(int(range_y)):
            if cur_left + overlap_range_width <= image.shape[2] and top + overlap_range_height <= image.shape[1]:
                patches_list.append(torchvision.transforms.functional.crop(image, top, cur_left, patch_height, patch_width))
                cur_left += (patch_width - overlap_range_width)
        top += (patch_height - overlap_range_height)
        cur_left = left

    return patches_list



# def crop_image_overlapping_parts(image, top, left, patch_height, patch_width, overlap):
#    """
#    the function crop the image to patches, according to the overlap. if the overlap is 0.5, then the patches will be
#    50% overlapping with eachother - meaning the next patch will start in the middle of the previous one.
#    """
#    patches = []
#    image_height, image_width = image.shape[1:]
#    for i in range(top, image_height - (image_height % patch_height) , int(patch_height * (1-overlap))):
#        for j in range(left, image_width - (image_width % patch_width), int(patch_width * (1-overlap))):
#            patch = torchvision.transforms.functional.crop(image, i, j, patch_height, patch_width)
#            patches.append(patch)
#    return patches

def create_patches_for_type(images_folder_path, patch_size, overlap, crop_start, n_rotations=None):
    '''
    The function get a folder with images, divides each image to patches, and take each patch and creates augmentations.
    :param images_folder: folder path as string
    :return: patches_all: tensor of patches tensors.
    '''
    patches_all = torch.tensor([])
    orig_images = []
    images_folder = PosixPath(images_folder_path)
    top, left = crop_start
    patch_height, patch_width = patch_size

    for image in images_folder.iterdir():
        if image.is_file():
            image_path = image.__fspath__()
            if "nd2" in image.suffix:
                nd = ND2Reader(image_path)
                img_arr = nd.get_frame(0)
                tensor_img = torch.tensor(np.array([np.int16(img_arr)]))
            else:
                try:
                    img = Image.open(image_path)
                except:
                    print("error - this file is not an image: ", image_path)
                    continue
                convert_tensor = transforms.ToTensor()
                tensor_img = convert_tensor(img)
                if torch.max(tensor_img) == 1:
                    tensor_img = tensor_img*255
            patches = crop_to_patches(tensor_img, top, left, patch_height, patch_width, overlap) #maia's function
            for patch in patches:
                augmentations = augmentation(patch, 2) # 2 - only non-interpolation augmentation
                patches_all = torch.cat((patches_all, augmentations))
            orig_images += ([image_path] * (len(patches)*4)) # 4 - number of augmentations

    return patches_all, orig_images

def show_patches(patches, n_rows, n_cols):
    '''
    The function get a tensor of patches and show them in a grid.
    '''
    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].imshow(patches[i*n_cols + j], cmap='gray')
    plt.show()

# def remove_outliers(patch):
#     '''
#     The function get a patch and remove outliers from it.
#     '''
#     patch = patch.astype(np.float32)
#     patch = patch - np.mean(patch)
#     patch = patch / np.std(patch)
#     patch = np.clip(patch, -3, 3)
#     patch = patch + 3
#     patch = patch / 6
#     patch = patch * 255
#     return patch

def remove_outliers(patch):
    '''
    the function get a patch and remove the outliers, according to the q1, q3.
    and them normalize it between 0-255.
    '''
    #q1 = np.quantile(patch, 0.01)
    q1 = np.quantile(patch, 0.25)
    #q3 = np.quantile(patch, 0.99)
    q3 = np.quantile(patch, 0.75)
    iqr = q3 - q1
    patch[patch > q3 + 1.5 * iqr] = q3 + 1.5 * iqr
    patch[patch < q1 - 1.5 * iqr] = q1 - 1.5 * iqr
    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    patch = patch * 255
    return patch

def save_patches(patches, output_folder):
    '''
    save the patches to a folder as jpg images.
    :param patches:
    :param output_folder:
    :return:
    '''
    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    for i, patch in enumerate(patches):
        patch = patch.squeeze()
        patch = patch.numpy()
        patch = remove_outliers(patch)
        patch = patch.astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save(output_folder / f"patch{i}.jpg")


if __name__ == '__main__':
    # with np.load('/data/GAN_project/diffusion_tries/samples/openai-2023-04-11-15-37-42-886600/samples_10x64x64x3.npz') as data:
    #    lst = data.files
    #    for item in data[lst[0]]:
    #        plt.imshow(item)
    #        plt.show()
    #
    # fldr = '/data/GAN_project/microtubules/onit/HR/patches_256x256_ol0.25_v2'
    # imgs_fldr = PosixPath(fldr)
    #
    # for image in imgs_fldr.iterdir():
    #     if image.is_file():
    #         image_path = image.__fspath__()
    #         if "nd2" in image.suffix:
    #             nd = ND2Reader(image_path)
    #             img_arr = nd.get_frame(0)
    #             tensor_img = torch.tensor(np.array([np.int16(img_arr)]))
    #         else:
    #             try:
    #                 img = Image.open(image_path)
    #             except:
    #                 print("error - this file is not an image: ", image_path)
    #                 continue
    #             convert_tensor = transforms.ToTensor()
    #             tensor_img = convert_tensor(img)
    #             if torch.max(tensor_img) == 1:
    #                 tensor_img = tensor_img * 255
    #             plt.imshow(img)


    #main_fldr = r"C:\Users\tav33\Documents\GAN_big\try_data\DL"
    main_fldr = "/data/GAN_project/tiff_files/good"
    patch_size = (256, 256)
    overlap = 0.25
    crop_start = (0,0)
    n_rotations = 2
    pixel_size = 106e-9
    #labels_path ='/data/GAN_project/labels_try_data.csv'
    #labels_path = r'C:\Users\tav33\Courses\ProjectGAN\labels_try_data.csv'
    output_folder = '/data/GAN_project/microtubules/shareloc/1305'
    patches, orig_images= create_patches_for_type(main_fldr, patch_size, overlap, crop_start, n_rotations)
    save_patches(patches, output_folder)
    # from micro_dataset import MicroscopePatchesDataset
    # patches_ds = MicroscopePatchesDataset(patches, orig_images, labels_path,pixel_size , "DNA")
    # #torch.save(patches_ds, '/data/GAN_project/scripts/patches.t')
    # torch.save(patches_ds, r'C:\Users\tav33\Documents\GAN_big\try_data\patches.t')
    # p1, l1 = patches_ds[0]
    # print(f"label:{l1}")
    # plt.imshow(p1, cmap='gray')
    # plt.show()


