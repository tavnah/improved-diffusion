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




def from_csv_to_tiff(T, img_size, scale, csv_file , pixel_size , dir_path):
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

def convert_to_tiff_csv_folder(csv_folder, output_folder, T, img_size, scale, csv_file , pixel_size , dir_path):

    return

def remove_outliers(patch):
    '''
    the function get a patch and remove the outliers, according to the q1, q3.
    and them normalize it between 0-255.
    '''
    q1 = np.quantile(patch, 0.05)
    q3 = np.quantile(patch, 0.95)
    iqr = q3 - q1
    patch[patch > q3 + 1.5 * iqr] = q3 + 1.5 * iqr
    patch[patch < q1 - 1.5 * iqr] = q1 - 1.5 * iqr
    patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
    patch = patch * 255
    return patch

image_no_out = remove_outliers(image)


image_uint8 = (image_no_out[0]).astype(np.uint8)
img = Image.fromarray(image_uint8)
img.save("/data/GAN_project/csv_files"+)

plt.imshow(image_no_out[0])
plt.show()

if __name__ = '__main__':
    img_size = 256  # low resolution image hight/ width
    scale = 10  # int(0.1609743 /106)  #resolution improvement
    pixel_size = 106  # nm #original image pixel size  (effective pixel size = 106 nm)
    T = 1  # tiff stack length - is it the frames?
    csv_file = 'alpha_tubulin_cell8'  # name of the csv file
    dir_path = "/data/GAN_project/csv_files"