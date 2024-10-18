import matplotlib.pyplot as plt
import os


def save_incrementing_plot(plot_data, base_filename, show_plot: bool, directory: str):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    i = 1
    # Check if the filename exists in the specified directory, increment if it does
    while os.path.exists(f"{directory}/{base_filename}_{i}.png"):
        i += 1

    # Save the plot with the incremented filename in the specified directory
    filename = f"{directory}/{base_filename}_{i}.png"

    # Plot and save the figure
    plt.imshow(plot_data, cmap='viridis', origin='upper', interpolation='nearest', extent=[0, 100, 0, 10], vmin=0,
               vmax=6)
    plt.grid(False)
    plt.axis('off')  # Turn off axis display for a cleaner image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Optionally display the plot
    if show_plot:
        plt.show()

    # Clear the current figure for the next plot
    plt.clf()

    print(f"Plot saved as {filename}")

    return filename