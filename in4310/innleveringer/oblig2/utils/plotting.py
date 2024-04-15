import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(losses, file_name):
    df = pd.DataFrame(losses, index=list(range(len(losses))), columns=['value'])
    s_factor = 0.7
    smooth = df.ewm(alpha=(1 - s_factor)).mean()

    plt.plot(df['value'], alpha=0.4, label='Actual loss value')
    plt.plot(smooth['value'], label='Smoothed out loss value')
    plt.legend()
    plt.title(f'Smoothing = {s_factor}')
    # Draw the grind in the background
    plt.grid(alpha=0.3)
    # Name the axes
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # Set the y-axis minimum to 0
    axes = plt.gca()  # gca is short for Get Current Axes
    axes.set_ylim([0.1, None])

    plt.savefig(f'{file_name}.png')
