import numpy as np
from perlin_noise import PerlinNoise
import scipy.signal
import matplotlib.pyplot as plt
from libtiff import TIFF
from scipy.ndimage import gaussian_filter
from skimage import color
from PIL import Image
from CARE.CSBDeep.csbdeep.io import save_training_data
import os
from perlin_numpy import generate_perlin_noise_2d


def get_LR_HR_couples(patches_path, output_folder, image_path_for_mu_sigma, NA, lamda, pixel_size, check_percenage=False, percentage_threshold=0.2, threshold = 0.4, ind=0):

    with np.load(patches_path) as data:
        #patches_amount = len(data.files)
        patches = data[data.files[0]]
        xy_shape = (patches.shape[0],patches.shape[1],patches.shape[2])
        X = np.zeros(xy_shape)
        Y = np.zeros(xy_shape)
        LR_path = output_folder + '/low/'
        HR_path = output_folder + '/high/'
        if not os.path.exists(LR_path):
            os.makedirs(LR_path)
        if not os.path.exists(HR_path):
            os.makedirs(HR_path)

        for i, patch in enumerate(patches):
            patch = color.rgb2gray(patch)
            patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
            patch = patch.astype('float32')
            if check_percenage:
                cur_percentage = np.sum(patch > threshold) / patch.size
                if cur_percentage > percentage_threshold:
                    continue

            #patch = ((patch*254) + 1).astype('uint8') # for poisson lamda not to be 0
            #plt.imshow(patch)
            #plt.show()
            LR_patch = LR_from_HR(patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=False)
            HR_image = Image.fromarray(patch)
            LR_image = Image.fromarray(LR_patch)
            LR_image.save(LR_path + str(i+ind) + '.tif')
            HR_image.save(HR_path + str(i+ind) + '.tif')
            #Y[i, :, :] = patch
            #X[i, :, :] = LR_patch
        #save_training_data(output_path, X, Y, 'CXY')

            #save LR_patch to output_folder/low, save patch to output_folder/high, using numpy




def LR_from_HR(HR_patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=False):

    threshold = 0.4
    threshold_patch = HR_patch.copy()
    threshold_patch[threshold_patch < threshold] = 0

    w, l = HR_patch.shape
    #add perlin noise
    #perlin_noise = PerlinNoise() #octaves = ?
    #perlin_noise_img = np.array([[perlin_noise([i / w, j / l]) for j in range(w)] for i in range(l)])
    perlin_noise = generate_perlin_noise_2d((w, l), (4, 4))
    perlin_noise = perlin_noise - np.min(perlin_noise)
    hr_perlin_noise = threshold_patch + (1/6)* perlin_noise

    #convolve with PSF
    psf_sigma = 0.25 * (lamda/pixel_size) / NA  #3 for trying because it too good
    hr_convolved = gaussian_filter(hr_perlin_noise, sigma=psf_sigma, truncate=4, mode='reflect')

    #add poisson noise
    poisson_noise = np.random.poisson(hr_convolved + 0.5)
    hr_convolved_poisson = hr_convolved + (1/40)*poisson_noise

    #add gaussian noise
    mu, std = calculate_mu_sigma_from_tiff(image_path_for_mu_sigma)
    #mu = mu / 255
    #std = std / 255
    gaussian_noise = np.random.normal(mu, std, hr_convolved_poisson.shape)
    hr_convolved_poisson_gaussian = hr_convolved_poisson + (1/40)*gaussian_noise

    #show figure of original patch, patch after perlin noise, patch after convolution, patch after poisson noise, patch after gaussian noise
    if show_progress:
        fig, axs = plt.subplots(1, 6, figsize=(17, 5))
        axs[0].imshow(HR_patch, cmap='gray')
        axs[0].set_title('original')
        axs[1].imshow(threshold_patch, cmap='gray')
        axs[1].set_title('after treshold')
        axs[2].imshow(hr_perlin_noise, cmap='gray')
        axs[2].set_title('perlin noise')
        axs[3].imshow(hr_convolved, cmap='gray')
        axs[3].set_title('convolved with PSF')
        axs[4].imshow(hr_convolved_poisson, cmap='gray')
        axs[4].set_title('poisson noise')
        axs[5].imshow(hr_convolved_poisson_gaussian, cmap='gray')
        axs[5].set_title('gaussian noise')
        plt.show()

    return hr_convolved_poisson_gaussian

def calculate_mu_sigma_from_tiff(image_path):
    tif = TIFF.open(image_path, mode='r')
    image = tif.read_image()
    #normalize image to 0 - 255
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #image = image * 255
    mu = np.mean(image)
    sigma = np.std(image)
    return mu, sigma

def show_ten_samples_of_LR_patches(npz1, npz2, label = 'X'):
    samples1 = np.load(npz1)
    samples2 = np.load(npz2)
    samples1 = samples1[label]
    samples2 = samples2[label]
    #use subplot
    fig, axs = plt.subplots(2, 5, figsize=(17, 5))
    for i in range(5):
        axs[0, i].imshow(samples1[i,0, :, :], cmap='gray')
        axs[1, i].imshow(samples2[i,0, :, :], cmap='gray')
    plt.show()

if __name__ == '__main__':

    #patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-05-02-08-03-36-590047/samples_10x256x256x3.npz'
    #patches_path = '/data/GAN_project/diffusion_tries/openai-2023-03-31-15-33-00-056364/samples_10x64x64x3.npz'
    patches_path ='/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-25-11-10-07-242376/samples_100x256x256x3.npz'

    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-29-07-34-28-426488/samples_1000x256x256x3.npz'
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-02-23-15-23-909604/samples_1000x256x256x3.npz'
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-37-10-595522/samples_900x256x256x3.npz'
    patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-38-33-595326/samples_200x256x256x3.npz'

    all_patches_paths = ['/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-29-07-34-28-426488/samples_1000x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-02-23-15-23-909604/samples_1000x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-37-10-595522/samples_900x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-38-33-595326/samples_200x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-06-07-51-19-275465/samples_900x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-06-07-51-19-275465/samples_600x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-06-07-51-19-275465/samples_300x256x256x3.npz']
    indxs = [1000,1000,900,200,900,600,300]
    image_path_for_mu_sigma = '/data/GAN_project/CARE/input_n_avg_10_all_no_data_area.tif'
    lamda = 510e-9# 488e-9 # m
    NA = 1.45#1.46
    scaling = 4
    pixel_size = 0.106e-6/scaling # m

    CARE_patches_path = '/data/GAN_project/CARE/Synthetic_tubulin_gfp/train_data/data_label.npz'
    train_patches_path = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000_no_dense_thresh0.1_higherSNR_lowerPSF/train_data.npz'
    show_ten_samples_of_LR_patches(CARE_patches_path,train_patches_path)
    # tif = TIFF.open('/data/GAN_project/CARE/real_data/alpha_tubulin_cell8_cropped.tif', mode='r')
    # HR_image = tif.read_image()
    # HR_image = color.rgb2gray(HR_image)
    # HR_image = (HR_image - np.min(HR_image)) / (np.max(HR_image) - np.min(HR_image))
    # HR_image = ((HR_image * 254) + 1).astype('uint8')
    # LR_image = LR_from_HR(HR_image, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=True)

    output_folder = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000_no_dense_thresh0.1_higherSNR_lowerPSF'

    for i in range(len(indxs)):
        if i ==0:
            cur_ind = 0
        else:
            cur_ind += indxs[i-1]
        cur_path = all_patches_paths[i]
        get_LR_HR_couples(cur_path, output_folder, image_path_for_mu_sigma, NA, lamda, pixel_size,
                          check_percenage=True, percentage_threshold=0.1, threshold = 0.6, ind=cur_ind)


    # with np.load(patches_path) as data:
    #    lst = data.files
    #    for patch in data[lst[0]]:
    #        patch = color.rgb2gray(patch)
    #        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    #        patch = ((patch*254) + 1).astype('uint8') # for poisson lamda not to be 0
    #        lr = LR_from_HR(patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=True)
    #        print('hey')