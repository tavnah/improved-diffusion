import numpy as np
from perlin_noise import PerlinNoise
import scipy.signal
import matplotlib.pyplot as plt
from libtiff import TIFF
from scipy.ndimage import gaussian_filter
from skimage import color
from CARE.CSBDeep.csbdeep.io import save_training_data



def get_LR_HR_couples(patches_path, output_path, image_path_for_mu_sigma, NA, lamda, pixel_size):
    with np.load(patches_path) as data:
        #patches_amount = len(data.files)
        patches = data[data.files[0]]
        xy_shape = (patches.shape[0],1,1,patches.shape[1],patches.shape[2])
        X = np.zeros(xy_shape)
        Y = np.zeros(xy_shape)

        for i, patch in enumerate(patches):
            patch = color.rgb2gray(patch)
            patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
            patch = ((patch*254) + 1).astype('uint8') # for poisson lamda not to be 0
            LR_patch = LR_from_HR(patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=False)
            Y[i, 0, 0, :, :] = patch
            X[i, 0, 0, :, :] = LR_patch
        save_training_data(output_path, X, Y, 'SCZYX')

            #save LR_patch to output_folder/low, save patch to output_folder/high, using numpy




def LR_from_HR(HR_patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=False):

    w, l = HR_patch.shape

    #add perlin noise
    perlin_noise = PerlinNoise() #octaves = ?
    perlin_noise_img = np.array([[perlin_noise([i / w, j / l]) for j in range(w)] for i in range(l)])
    hr_perlin_noise = HR_patch + perlin_noise_img

    #convolve with PSF
    psf_sigma = 0.25 * (lamda/pixel_size) / NA
    hr_convolved = gaussian_filter(hr_perlin_noise, sigma=psf_sigma, truncate=3, mode='reflect')

    #add poisson noise
    hr_convolved_poisson = np.random.poisson(hr_convolved)

    #add gaussian noise
    mu, std = calculate_mu_sigma_from_tiff(image_path_for_mu_sigma)
    gaussian_noise = np.random.normal(mu, std, hr_convolved_poisson.shape)
    hr_convolved_poisson_gaussian = hr_convolved_poisson + (1/5)*gaussian_noise

    #show figure of original patch, patch after perlin noise, patch after convolution, patch after poisson noise, patch after gaussian noise
    if show_progress:
        fig, axs = plt.subplots(1, 5, figsize=(17, 5))
        axs[0].imshow(HR_patch, cmap='gray')
        axs[0].set_title('original')
        axs[1].imshow(hr_perlin_noise, cmap='gray')
        axs[1].set_title('perlin noise')
        axs[2].imshow(hr_convolved, cmap='gray')
        axs[2].set_title('convolved with PSF')
        axs[3].imshow(hr_convolved_poisson, cmap='gray')
        axs[3].set_title('poisson noise')
        axs[4].imshow(hr_convolved_poisson_gaussian, cmap='gray')
        axs[4].set_title('gaussian noise')
        plt.show()

    return hr_convolved_poisson_gaussian

def calculate_mu_sigma_from_tiff(image_path):
    tif = TIFF.open(image_path, mode='r')
    image = tif.read_image()
    #normalize image to 0 - 255
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    mu = np.mean(image)
    sigma = np.std(image)
    return mu, sigma

if __name__ == '__main__':
    #patches_path = '/data/GAN_project/diffusion_tries/samples/openai-2023-05-02-08-03-36-590047/samples_10x256x256x3.npz'
    patches_path = '/data/GAN_project/diffusion_tries/openai-2023-03-31-15-33-00-056364/samples_10x64x64x3.npz'
    image_path_for_mu_sigma = '/data/GAN_project/CARE/input_n_avg_10_all_no_data_area.tif'
    lamda = 488e-9 # m
    NA = 1.46
    pixel_size = 0.11e-6 # m
    output_folder = '/data/GAN_project/CARE/simulated_LR'
    output_path = output_folder + '/train_data.npz'
    get_LR_HR_couples(patches_path, output_path, image_path_for_mu_sigma, NA, lamda, pixel_size)


    # with np.load(patches_path) as data:
    #    lst = data.files
    #    for patch in data[lst[0]]:
    #        patch = color.rgb2gray(patch)
    #        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    #        patch = ((patch*254) + 1).astype('uint8') # for poisson lamda not to be 0
    #        lr = LR_from_HR(patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=True)
    #        print('hey')