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
from skimage.transform import resize



def get_LR_HR_couples(patches_path, output_folder, image_path_for_mu_sigma, NA, lamda, pixel_size, check_percenage=False, percentage_threshold=0.2, threshold = 0.4, ind=0, use_smooth=False, use_threshold=False):

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
            if patch.std() < 0.1: #good for mito_v2
                continue

            #patch = ((patch*254) + 1).astype('uint8') # for poisson lamda not to be 0
            #plt.imshow(patch)
            #plt.show()
            LR_patch = LR_from_HR(patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=False, use_threshold=use_threshold, use_smooth=use_smooth)
            HR_image = Image.fromarray(patch)
            LR_image = Image.fromarray(LR_patch)
            LR_image.save(LR_path + str(i+ind) + '.tif')
            HR_image.save(HR_path + str(i+ind) + '.tif')
            #Y[i, :, :] = patch
            #X[i, :, :] = LR_patch
        #save_training_data(output_path, X, Y, 'CXY')

            #save LR_patch to output_folder/low, save patch to output_folder/high, using numpy




def LR_from_HR(HR_patch, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=False, use_threshold=False, use_smooth=False):

    threshold = 0.2
    cur_patch = HR_patch
    if use_threshold:
        threshold_patch = HR_patch.copy()
        threshold_patch[threshold_patch < threshold] = 0 # FOR MICROTUBULES
        cur_patch = threshold_patch

    #smooth image with gaussian filter - for mito try1
    if use_smooth:
        smoothed_patch = gaussian_filter(cur_patch, sigma=1, truncate=4, mode='reflect')
        cur_patch = smoothed_patch

    w, l = cur_patch.shape
    #add perlin noise
    #perlin_noise = PerlinNoise() #octaves = ?
    #perlin_noise_img = np.array([[perlin_noise([i / w, j / l]) for j in range(w)] for i in range(l)])
    new_w = w - (w%16)
    new_l = l - (l%16)
    patch = cur_patch[:new_w, :new_l]
    perlin_noise = generate_perlin_noise_2d((new_w, new_l), (4, 4))
    perlin_noise = perlin_noise - np.min(perlin_noise)
    hr_perlin_noise = patch + (1/12)* perlin_noise

    #convolve with PSF
    psf_sigma = 0.25 * (lamda/pixel_size) / NA  #3 for trying because it too good
    hr_convolved = gaussian_filter(hr_perlin_noise, sigma=psf_sigma, truncate=4, mode='reflect')

    #add poisson noise
    poisson_noise = np.random.poisson(hr_convolved + 0.5)
    hr_convolved_poisson = hr_convolved + (1/20)*poisson_noise#(1/40)*poisson_noise

    #add gaussian noise
    mu, std = calculate_mu_sigma_from_tiff(image_path_for_mu_sigma)
    #mu = mu / 255
    #std = std / 255
    gaussian_noise = np.random.normal(mu, std, hr_convolved_poisson.shape)
    hr_convolved_poisson_gaussian = hr_convolved_poisson + (1/20)*gaussian_noise#(1/40)*gaussian_noise

    #show figure of original patch, patch after perlin noise, patch after convolution, patch after poisson noise, patch after gaussian noise
    if show_progress:
        fig, axs = plt.subplots(1, 6, figsize=(17, 5))
        axs[0].imshow(HR_patch, cmap='gray')
        axs[0].set_title('original')
        axs[1].imshow(cur_patch, cmap='gray')
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
    # patches_path ='/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-25-11-10-07-242376/samples_100x256x256x3.npz'
    #
    # patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-29-07-34-28-426488/samples_1000x256x256x3.npz'
    # patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-02-23-15-23-909604/samples_1000x256x256x3.npz'
    # patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-37-10-595522/samples_900x256x256x3.npz'
    # patches_path = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-38-33-595326/samples_200x256x256x3.npz'

    all_patches_paths_microtub = ['/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-05-29-07-34-28-426488/samples_1000x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-02-23-15-23-909604/samples_1000x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-37-10-595522/samples_900x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-04-09-38-33-595326/samples_200x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-06-07-51-19-275465/samples_900x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-06-07-51-19-275465/samples_600x256x256x3.npz',
                         '/data/GAN_project/diffusion_tries/samples/shareloc/1305/openai-2023-06-06-07-51-19-275465/samples_300x256x256x3.npz']
    indxs = [1000,1000,900,200,900,600,300]

    image_path_for_mu_sigma = '/data/GAN_project/CARE/input_n_avg_10_all_no_data_area.tif' #microtubules
    lamda = 660e-9  # 510e-9 for microtubules, 660e-9 for mitochondriaold
    NA = 1.46#1.46 for microtubules, 1.49 for mitochondriaold
    scaling = 4
    pixel_size = 0.106e-6/scaling # m

    # CARE_patches_path = '/data/GAN_project/CARE/Synthetic_tubulin_gfp/train_data/data_label.npz'
    # train_patches_path = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000_no_dense_thresh0.1_higherSNR_lowerPSF/train_data.npz'
    # show_ten_samples_of_LR_patches(CARE_patches_path,train_patches_path)
    microtubules = False
    mitochondria = False
    one_image = True
    # ======= microtubules =======
    output_folder = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000_thresh0.1_higherSNR_pix0.08_lamb510_NA1.46'
    if microtubules:
        for i in range(len(indxs)):
            if i ==0:
                cur_ind = 0
            else:
                cur_ind += indxs[i-1]
            cur_path = all_patches_paths_microtub[i]
            get_LR_HR_couples(cur_path, output_folder, image_path_for_mu_sigma, NA, lamda, pixel_size,
                              check_percenage=True, percentage_threshold=0.1, threshold = 0.6, ind=cur_ind,
                              use_threshold=True)

    # ======= mitochondria =======
    #output_folder_mito = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/mitochondria/thresh0.18_pix0.106_lamb660_NA1.49_smoothed_noperlin'
    output_folder_mito = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_mito/no_thresh_pix0.106_lamb510_NA1.46'
    #patches_mito_folder = '/data/GAN_project/diffusion_tries/samples/mitochondria/1106'
    patches_mito_folder = '/data/GAN_project/diffusion_tries/samples/mitochondria/shareloc/3006'
    cur_ind = 0
    if mitochondria:
        for folder in os.listdir(patches_mito_folder):
            if folder.startswith('openai'):
                for file in os.listdir(os.path.join(patches_mito_folder, folder)):
                    if file.endswith('.npz'):
                        patches_path = os.path.join(patches_mito_folder, folder, file)
                        # old mito
                        # get_LR_HR_couples(patches_path, output_folder_mito, image_path_for_mu_sigma, NA, lamda, pixel_size,
                        #                   check_percenage=True, percentage_threshold=0.18, threshold = 0.6, ind = cur_ind, use_smooth=True, use_threshold=True)
                        get_LR_HR_couples(patches_path, output_folder_mito, image_path_for_mu_sigma, NA, lamda,
                                          pixel_size,
                                          check_percenage=False, percentage_threshold=0.18, threshold=0.6, ind=cur_ind)
                        cur_ind += int(file.split('_')[1].split('x')[0])

    if one_image:
        hr_im_path = '/data/GAN_project/mitochondria/onit/COS7_tom20_647_unfiltered003-second_STORM.jpg'
        output_path = '/data/GAN_project/test_imgs/onit_mit/2/COS7_tom20_647_unfiltered003-second_STORM_LR.tiff'

        hr_im = Image.open(hr_im_path)
        hr_np = np.asarray(hr_im)
        lr_np = LR_from_HR(hr_np, image_path_for_mu_sigma, NA, lamda, pixel_size, show_progress=True)
        new_size = (int(lr_np.shape[0] // 4), int(lr_np.shape[1] // 4))
        lr_np = resize(lr_np, new_size)
        lr_im = Image.fromarray(lr_np)
        lr_im.save(output_path)
