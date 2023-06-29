import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
from sewar.full_ref import msssim
from skimage import exposure
import scipy

#matplotlib.use('TkAgg')

def norm_01(img):
    return (img - img.min()) / (img.max() - img.min())

def norm_percentile(img, pcnt):
    return norm_01(saturte_percentile(img, pcnt))

def saturte_percentile(img, pcnt):
    tmp_img = np.copy(img)
    tmp_img[np.where(img > np.percentile(img, pcnt))] = np.percentile(img, pcnt)
    return tmp_img

def binarize_percentile(img, pcnt):
    tmp_img = np.zeros_like(img)
    tmp_img[np.where(img > np.percentile(img, pcnt))] = 255
    tmp_img[np.where(img <= np.percentile(img, pcnt))] = 0
    return tmp_img

def register_gt(img1, img2):
    # Convert to grayscale.
    img1 = (255 * img1).astype(np.uint8)
    img2 = (255 * img2).astype(np.uint8)

    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1, homography, (width, height))
    return transformed_img
def compare_imgs_MS_SSIM(img1, img2):
    #ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    img1 = normalize(img1)
    img2 = normalize(img2)
    img1_int = (img1 * 255).astype(np.uint8)
    img2_int = (img2 * 255).astype(np.uint8)
    return msssim(img1_int, img2_int)

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
def compare_MSE(img, gt):
    mse1 = np.mean((img - gt) ** 2)
    return mse1

def find_best_shift(img1, img2):
    '''
    using cross correlation to find the best shift between two images and shift img2 accordingly
    '''
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    correlation = scipy.signal.fftconvolve(img1, img2[::-1, ::-1], mode='same')

    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift_y, shift_x = img1.shape[0] // 2 - y, img1.shape[1] // 2 - x
    shifted_img1 = scipy.ndimage.shift(img1, (shift_y, shift_x), cval=0)
    return shifted_img1

def create_two_overlaps(img1, img2, gt):
    img1_gt_ol = np.zeros((img1.shape[0], img1.shape[1], 3))
    img1_gt_ol[:, :, 0] = img1
    img1_gt_ol[:, :, 1] = gt
    img2_gt_ol = np.zeros((img2.shape[0], img2.shape[1], 3))
    img2_gt_ol[:, :, 0] = img2
    img2_gt_ol[:, :, 1] = gt
    return img1_gt_ol, img2_gt_ol

#res1_dir_path = 'C:/Users/alonsaguy/PHD/GAN_simulator_project/output_orig'
#res1_dir_path = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/output_orig'
res1_dir_path = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/poster'

#res1_dir_path = r'C:\Users\tav33\Desktop\test_imgs_1000\shareloc_MT3D\output_orig_1000'
zoom7 = True
for filename in os.listdir(res1_dir_path):
    if '7.npz' in filename:
        shift = 20
        res = np.load(os.path.join(res1_dir_path, filename))
        care_recon = norm_percentile(res['care'][shift:-shift, shift:-shift, 0], 95)
        gt = norm_01(res['care'][shift:-shift, shift:-shift, 1])
        our_recon = norm_percentile(res['ours'][shift:-shift, shift:-shift, 0], 95)

        final_shape = care_recon.shape
        gt_registered = np.array(norm_01(register_gt(gt, our_recon)), dtype=float)

        if zoom7 == True:
            our_zoom = our_recon[:75,50:200]
            gt_zoom = gt[:75,50:200]
            care_zoom = care_recon[:75,50:200]
            gt_zoom_moved = find_best_shift(gt_zoom, our_zoom)
            our_ol, care_ol = create_two_overlaps(our_zoom,care_zoom, gt_zoom_moved)
            print("heyhey")

        print("file_name: ", filename)
        mse_ours = compare_MSE(our_recon, gt_registered)
        mse_care = compare_MSE(care_recon, gt_registered)
        mssim_ours = compare_imgs_MS_SSIM(our_recon, gt_registered)
        mssim_care = compare_imgs_MS_SSIM(care_recon, gt_registered)
        print('MSE for ours: ', mse_ours)
        print('MSE for CARE: ', mse_care)
        if mse_ours < mse_care:
            print('ours is better MSE')
        else:
            print('care is better MSE')
        print('MS-SSIM for ours: ', mssim_ours)
        print('MS-SSIM for CARE: ', mssim_care)
        if mssim_ours > mssim_care:
            print('ours is better MS-SSIM')
        else:
            print('care is better MS-SSIM')

        ax1 = plt.subplot(231)
        plt.imshow(care_recon)
        plt.title('care recon')
        plt.subplot(232, sharex=ax1, sharey=ax1)
        plt.imshow(our_recon)
        plt.title('our recon')
        plt.subplot(233, sharex=ax1, sharey=ax1)
        plt.imshow(gt_registered)
        plt.title('gt recon (SMLM)')
        plt.subplot(234, sharex=ax1, sharey=ax1)
        plt.imshow(norm_01(gt_registered * care_recon))
        plt.title("gt * CARE")
        plt.subplot(235, sharex=ax1, sharey=ax1)
        plt.imshow(norm_01(gt_registered * our_recon))
        plt.title("gt * our")
        plt.subplot(236, sharex=ax1, sharey=ax1)
        plt.imshow(np.concatenate([care_recon[:, :, None], our_recon[:, :, None], gt_registered[:, :, None]], axis=2))
        plt.title('overlap care (red) ours (green) gt (blue)')
        plt.show()