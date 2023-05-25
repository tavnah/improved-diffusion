import pandas as pd
import matplotlib.pyplot as plt
from scripts.image_sample import main as sample_main
from pathlib import PosixPath
import numpy as np

def show_loss(progress_csv_path):
    progress_df = pd.read_csv(progress_csv_path)
    loss = progress_df['loss']#.values
    steps = progress_df['step']#.values
    loss = loss[2:]
    steps = steps[2:]
    plt.plot(steps, loss)
    plt.show()

def check_sampled_images_along_training(model_folder_path, output_folder, step, start, end):

    folder = PosixPath(model_folder_path)
    for i in range(start, end, step):
        current_iter = str(i).zfill(6)
        current_ema = f'ema_0.9999_{current_iter}.pt'
        ema_path = model_folder_path + '/' + current_ema
        output_path = output_folder + '/' + current_iter
        sample_main(ema_path, output_path)

def show_sampled_image_per_iter(folder_path, output_path):
    main_folder = PosixPath(folder_path)
    for folder in main_folder.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            for file in folder.iterdir():
                if file.name.endswith("npz"):
                    npz_path = file.__fspath__()
                    with np.load(npz_path) as data:
                        lst = data.files
                        total_patches_num = len(data[lst[0]])
                        rows_num = 2
                        cols_num = 5
                        plt.figure(figsize=(30, 10))
                        for i, patch in enumerate(data[lst[0]]):
                            plt.subplot(rows_num, cols_num, i + 1)
                            plt.imshow(patch)
                            plt.title(f'patch {i}')
                        plt.suptitle(folder_name)
                        plt.savefig(output_path + '/' + folder_name + '.png')


if __name__ == '__main__':
    #model_folder_path = '/data/GAN_project/diffusion_tries/microtubules/tav/alpha_tubulin_scale_4/openai-2023-05-18-23-45-52-473911'
    #progress_csv_path = model_folder_path + '/progress.csv'
    #show_loss(progress_csv_path)

    model_folder_path = '/data/GAN_project/diffusion_tries/microtubules/tav/alpha_tubulin_scale_4/openai-2023-05-19-09-20-28-014773'
    output_folder = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/mag4_diff3000'
    check_sampled_images_along_training(model_folder_path, output_folder, step=2000, start=30000, end=46000 )
    output_plots = '/data/GAN_project/diffusion_tries/samples/shareloc/1305/mag4_diff3000/plots'
    show_sampled_image_per_iter(output_folder, output_plots)

    #progress_csv_path = '/data/GAN_project/diffusion_tries/mitochondria/tav/openai-2023-04-25-18-02-13-010307/progress.csv'
    #progress_csv_path = r'/data/GAN_project/diffusion_tries/microtubules/tav/alpha_tubulin_scale_4/openai-2023-05-18-23-45-52-473911/progress.csv'
    #progress_csv_path = '/data/GAN_project/diffusion_tries/microtubules/tav/alpha_tubulin_scale_4/openai-2023-05-19-09-20-28-014773/progress.csv'
    model_folder_path = '/data/GAN_project/diffusion_tries/microtubules/tav/alpha_tubulin_scale_4/openai-2023-05-18-23-45-52-473911'
    progress_csv_path = model_folder_path + '/progress.csv'
    show_loss(progress_csv_path)