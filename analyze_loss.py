import pandas as pd
import matplotlib.pyplot as plt

def show_loss(progress_csv_path):
    progress_df = pd.read_csv(progress_csv_path)
    loss = progress_df['loss']
    steps = progress_df['step']
    plt.plot(steps, loss)
    plt.show()

if __name__ == '__main__':
    progress_csv_path = '/data/GAN_project/diffusion_tries/mitochondria/tav/openai-2023-04-23-15-28-55-434008/progress.csv'
    show_loss(progress_csv_path)