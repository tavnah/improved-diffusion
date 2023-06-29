# inputs:
# img_size = low resolution image hight/ width
# scale = resolution improvement
# pixel size = original image pixel size
# T = tiff stack length
# csv_file = name of the csv file
# dir_path = path to csv file
import numpy as np
import os
from csv import reader
from PIL import Image
import matplotlib.pyplot as plt




def from_csv_to_tiff(T, img_size, scale, csv_file , pixel_size , dir_path, output_folder):
    image = np.zeros([T, img_size * scale, img_size * scale])
    i = 0
    with open(os.path.join(dir_path, '{}.csv'.format(csv_file)), 'r') as read_obj:
        csv_reader = reader(read_obj)
        next(csv_reader,None)
        for my_data in csv_reader:

            #if(i == 0 or int(float(my_data[0])) > T):
            #    i += 1
            #    continue
            # check that indices do not exceed image size
            if my_data[2] == 'nan' or my_data[3] == 'nan':
                continue
            x = float(my_data[2])*scale/pixel_size
            y = float(my_data[3])*scale/pixel_size

            if(int(y) < 0 or
                    int(x) < 0 or
                    int(y) >= img_size*scale or
                    int(x) >= img_size*scale):
                continue
            image[0,
                  int(y),
                  int(x)] += 1
            i += 1
    image_no_out = remove_outliers(image)
    image_new = image_no_out[0].astype('float32')
    im = Image.fromarray(image_new)
    im.save(os.path.join(output_folder, '{}.tiff'.format(csv_file)))


def convert_to_tiff_csv_folder(dir_path, output_folder, T, img_size, scale , pixel_size ):
    '''
    the function get a folder with csv files and convert them to tiff files, and save each file in output folder.
    '''
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            file_name = file.split('.')[0]
            from_csv_to_tiff(T, img_size, scale, file_name , pixel_size , dir_path, output_folder)

def remove_outliers(patch):
    '''
    the function get a patch and remove the outliers, according to the q1, q3.
    and them normalize it between 0-255.
    '''
    q1 = np.quantile(patch, 0.01)
    q3 = np.quantile(patch, 0.99)
    iqr = q3 - q1
    patch[patch > q3 + 1.5 * iqr] = q3 + 1.5 * iqr
    patch[patch < q1 - 1.5 * iqr] = q1 - 1.5 * iqr
    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    #patch = patch * 255
    return patch

def from_np_to_tif(np_array, output_folder, file_name):
    ''' the function get a numpy array and save it as tiff file'''
    im = Image.fromarray(np_array)
    im.save(os.path.join(output_folder, '{}.tiff'.format(file_name)))

def folder_of_npz_to_tiff(npz_folder, output_folder):
    for file in os.listdir(npz_folder):
        if file.endswith(".npz"):
            file_name = file.split('.')[0]
            npz = np.load(os.path.join(npz_folder, file))
            care3 = npz['care']
            care = care3[:,:,0]
            hr = care3[:,:,1]
            ours3 = npz['ours']
            ours = ours3[:,:,0]
            from_np_to_tif(care, output_folder, file_name + '_care')
            from_np_to_tif(hr, output_folder, file_name + '_gt')
            from_np_to_tif(ours, output_folder, file_name + '_ours')

if __name__ == '__main__':
    img_size = 256  # low resolution image hight/ width
    scale = 4  # int(0.1609743 /106)  #resolution improvement
    pixel_size = 106  # nm #original image pixel size  (effective pixel size = 106 nm)
    T = 1  # tiff stack length - is it the frames?
    #csv_file = 'alpha_tubulin_cell8'  # name of the csv file
    #dir_path = "/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/3"
    #output_folder = "/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/3"
    dir_path = "/data/GAN_project/test_imgs/shareloc2/5"
    output_folder = "/data/GAN_project/test_imgs/shareloc2/5"

    dir_path = '/data/GAN_project/test_imgs/shareloc_mit'
    output_folder = '/data/GAN_project/test_imgs/shareloc_mit'
    convert_to_tiff_csv_folder(dir_path, output_folder, T, img_size, scale, pixel_size)
    npz_folder = "/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/output_orig_1000"
    output_folder = "/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/output_orig_1000"
    folder_of_npz_to_tiff(npz_folder, output_folder)
