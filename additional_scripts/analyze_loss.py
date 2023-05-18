import pandas as pd
import matplotlib.pyplot as plt

def show_loss(progress_csv_path):
    progress_df = pd.read_csv(progress_csv_path)
    loss = progress_df['loss']#.values
    steps = progress_df['step']#.values
    loss = loss[2:]
    steps = steps[2:]
    plt.plot(steps, loss)
    plt.show()


if __name__ == '__main__':
    progress_csv_path = '/data/GAN_project/diffusion_tries/mitochondria/tav/openai-2023-04-25-18-02-13-010307/progress.csv'
    progress_csv_path = r'C:\Users\tav33\Courses\ProjectGAN\model\progress.csv'
    show_loss(progress_csv_path)