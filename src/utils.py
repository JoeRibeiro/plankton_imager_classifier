import os
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from fastai.vision.all import *
from fastai.interpret import ClassificationInterpretation
from memory_profiler import profile
from pathlib import Path
from PIL import Image


""" Functions used in training """
def analyze_tif_files(main_directory):
    """Merged function from src/archive/Preprocess_02_v01_Describe_train_data.ipynb"""
    subdirectory_counts = {}
    tif_count = 0  # Initialize total count

    for subdir, _, files in os.walk(main_directory):
        count = len([file for file in files if file.endswith('.tif')])
        tif_count += count  # Add to total count

        # For subdirectory counts, exclude the main directory itself
        if count > 0 and subdir != main_directory:
            subdirectory_name = os.path.basename(subdir)
            subdirectory_counts[subdirectory_name] = count

    # Print the total count
    print(f"Total number of images in {main_directory}: {tif_count:,}")

    # Create DataFrame from subdirectory counts and sort
    df = pd.DataFrame(list(subdirectory_counts.items()), columns=['Label', 'Count'])
    df = df.sort_values(by='Count', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))

    plt.bar(df['Label'], df['Count'], color='skyblue')

    plt.yscale('log')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Count of images per Class')
    plt.xticks(rotation=90)

    plt.tight_layout()
    filepath = "doc/dataset_description.png"
    plt.savefig(filepath)
    # plt.show()

    print(f"[INFO] Printed figure describing class distribution in training dataset at {filepath}")

def plot_category_examples(data_path, output_path='doc/train_example.png'):
    """
    Plot one example image from each category in a 7x7 grid layout.

    Args:
        data_path: Path to the directory containing category subdirectories
        output_path: Path to save the output visualization
    """
    # 1. Find all category directories and sort them alphabetically
    main_path = Path(data_path)
    category_dirs = sorted([d for d in main_path.iterdir() if d.is_dir()])

    # We expect 49 categories, but handle cases where there are fewer
    grid_size = 7
    total_categories = grid_size * grid_size  # 49

    # If there are more than 49 categories, we'll only plot the first 49
    category_dirs = category_dirs[:total_categories]

    # Create figure with appropriate size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    # Flatten the 2D array of axes for easy iteration
    axes = axes.ravel()

    # 2. For each category, plot an example image
    for i, category_dir in enumerate(category_dirs):
        # Find the first image in this category directory
        images = sorted(category_dir.glob('*.tif'))  # Sort to ensure consistent selection

        if images:  # If there are any images in this category
            img_path = images[0]
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(category_dir.name, fontsize=8)
                axes[i].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # If image can't be loaded, just show the title
                axes[i].set_title(f"{category_dir.name}\n(Error loading image)", fontsize=8)
                axes[i].axis('off')
                # Add empty image placeholder
                axes[i].imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        else:
            # If no images in category, just show the title with empty space
            axes[i].set_title(f"{category_dir.name}\n(No images)", fontsize=8)
            axes[i].axis('off')
            axes[i].imshow(np.zeros((100, 100, 3), dtype=np.uint8))  # Empty RGB image

    # Hide any unused subplots (if there are fewer than 49 categories)
    for j in range(len(category_dirs), len(axes)):
        axes[j].axis('off')

    # Adjust layout to minimize overlap
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.5, hspace=0.8)  # Adjust spacing between subplots

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    print(f"[INFO] Saved category visualization to {output_path}")
    plt.close()

def save_data_visualizations(dls, images_root):
    """
    Save various data visualizations to files in a training subdirectory.

    Args:
        dls: DataLoader object containing the data
        base_path: Base directory to save visualizations (default: 'doc')
    """

    # 1. General data overview with transformations
    fig = dls.show_batch(nrows=2, ncols=5, unique=True)
    plt.suptitle('General Data Overview with Transformations (Training + Validation)', y=1.02)
    plt.savefig(os.path.join(images_root, '01_data_overview.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)

    # 2. Training set with augmentations (most important visualization)
    fig = dls.train.show_batch(max_n=2, nrows=1, unique=True)
    plt.suptitle('Training Data with Augmentations', y=1.02)
    plt.savefig(os.path.join(images_root, '02_train_augmentation.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)

    # 3. Validation set visualization
    fig = dls.valid.show_batch(max_n=10, nrows=1)
    plt.suptitle('Validation Data Sample', y=1.02)
    plt.savefig(os.path.join(images_root, '03_validation_data.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)

    # 4. Simple batch visualization
    fig = dls.show_batch(nrows=1, ncols=5)
    plt.suptitle('Random Data Sample', y=1.02)
    plt.savefig(os.path.join(images_root, '04_random_sample.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)

    print(f"[INFO] All training/validation examples saved to {images_root}.")

def save_evaluation_visualizations(learn, images_root):
    """
    Save evaluation visualizations to the specified directory.

    Args:
        learn: FastAI Learner object
        images_root: Path to directory where images should be saved
    """
    # Create output directory if it doesn't exist
    Path(images_root).mkdir(parents=True, exist_ok=True)

    # Create interpretation object
    interp = ClassificationInterpretation.from_learner(learn)

    # 1. Confusion matrix
    plt.figure(figsize=(20, 20))
    interp.plot_confusion_matrix(figsize=(20, 20))
    plt.savefig(os.path.join(images_root, 'confusion_matrix.png'), bbox_inches='tight', dpi=100)
    # print(f"[INFO] Saved confusion matrix to {cm_path}")
    plt.close()

    # 2. Top losses
    plt.figure()
    interp.plot_top_losses(20, nrows=20)
    plt.savefig(os.path.join(images_root, 'top_losses.png'), bbox_inches='tight', dpi=100)
    # print(f"[INFO] Saved top losses visualization to {losses_path}")
    plt.close()

    # 3. Most confused
    plt.figure()
    most_confused = interp.most_confused(min_val=2)
    if most_confused:
        # For most_confused, we need to save the returned figure
        plt.savefig(os.path.join(images_root, 'most_confused.png'), bbox_inches='tight', dpi=100)
    plt.close()

    # 4. Show results (multiple examples)
    plt.figure(figsize=(10, 10))
    learn.show_results()
    plt.savefig(os.path.join(images_root, 'results_examples.png'), bbox_inches='tight', dpi=100)
    plt.close()

    print(f"[INFO] Saved output to {images_root}")

""" Functions used in inference """