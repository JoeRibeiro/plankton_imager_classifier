import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from fastai.vision.all import *
from fastai.interpret import ClassificationInterpretation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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

"""Functions used in model evaluation """
def plot_class_distribution(images_root, sorted_labels, dpi=300):
    """
    Plot a histogram of class distribution and save it to the specified directory.

    Parameters:
    - images_root: Path to save the output image.
    - sorted_labels: Pandas Series containing the class labels.
    - dpi: Dots per inch for the saved image (default: 300).
    """
    # Create the directory if it doesn't exist
    os.makedirs(images_root, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sorted_labels.value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Classes in test-set', fontsize=16)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    # Save the plot
    output_path = os.path.join(images_root, 'class_distribution.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved class distribution plot to {output_path}")

def plot_confusion_matrix(images_root, filename, merged_df, y_true, y_pred, dpi=300):
    """
    Plot a confusion matrix heatmap and save it to the specified directory.

    Parameters:
    - images_root: Path to save the output image.
    - merged_df: DataFrame containing the label and pred_label columns.
    - y_true: True labels.
    - y_pred: Predicted labels.
    - dpi: Dots per inch for the saved image (default: 300).
    """
    # Create the directory if it doesn't exist
    os.makedirs(images_root, exist_ok=True)

    # Define class labels
    unique_classes = sorted(merged_df['label'].unique())

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Create a custom annotations array where zeroes are replaced with empty strings
    annotations = np.where(cm == 0, '', cm)

    # Create the plot
    plt.figure(figsize=(20, 14))
    sns.heatmap(cm, annot=annotations, fmt='',
                cmap='Blues',
                cbar=True,
                xticklabels=unique_classes,
                yticklabels=unique_classes,
                annot_kws={"size": 10},
                linewidths=0.5,
                linecolor='white')

    plt.xlabel('Predicted labels', fontsize=14)
    plt.ylabel('True labels', fontsize=14)
    plt.title(f'Confusion Matrix ({filename}', fontsize=16)
    plt.xticks(fontsize=12, rotation=-90)
    plt.yticks(fontsize=12)

    # Save the plot
    output_path = os.path.join(images_root, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved confusion matrix plot to {output_path}")

def plot_classification_metrics(images_root, merged_df, dpi=300):
    """
    Plot precision, recall, and F1 scores by class and save to the specified directory.

    Parameters:
    - images_root: Path to save the output image.
    - merged_df: DataFrame containing the label and pred_label columns.
    - dpi: Dots per inch for the saved image (default: 300).
    """
    # Create the directory if it doesn't exist
    os.makedirs(images_root, exist_ok=True)

    # Font size variable
    font_size = 18

    # Set global font size
    plt.rcParams.update({'font.size': font_size})

    # Get unique classes sorted
    unique_classes = sorted(merged_df['label'].unique())

    # Initialize dictionaries for metrics
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    # Calculate metrics
    for class_label in unique_classes:
        precision = precision_score(merged_df['label'], merged_df['pred_label'], labels=[class_label], average='macro', zero_division=0)
        recall = recall_score(merged_df['label'], merged_df['pred_label'], labels=[class_label], average='macro', zero_division=0)
        f1 = f1_score(merged_df['label'], merged_df['pred_label'], labels=[class_label], average='macro', zero_division=0)
        
        precision_scores[class_label] = precision
        recall_scores[class_label] = recall
        f1_scores[class_label] = f1

    # Convert to lists for plotting
    sorted_classes = sorted(precision_scores.keys())
    sorted_precisions = [precision_scores[class_label] for class_label in sorted_classes]
    sorted_recalls = [recall_scores[class_label] for class_label in sorted_classes]
    sorted_f1s = [f1_scores[class_label] for class_label in sorted_classes]

    # Calculate mean values (with protection against empty lists)
    if sorted_precisions:
        mean_precision = sum(sorted_precisions) / len(sorted_precisions)
        mean_recall = sum(sorted_recalls) / len(sorted_recalls)
        mean_f1 = sum(sorted_f1s) / len(sorted_f1s)
    else:
        mean_precision = mean_recall = mean_f1 = 0

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Plot Precision
    axes[0].barh([str(class_label) for class_label in sorted_classes], sorted_precisions, color='skyblue')
    axes[0].set_xlabel('Precision [%]')
    axes[0].set_title(f'Precision (μ={mean_precision*100:.1f})%')
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    for i, class_label in enumerate(sorted_classes):
        axes[0].text(sorted_precisions[i] + 0.01, i, f'{sorted_precisions[i]*100:.0f}', va='center', ha='left')

    # Plot Recall
    axes[1].barh([str(class_label) for class_label in sorted_classes], sorted_recalls, color='salmon')
    axes[1].set_xlabel('Recall [%]')
    axes[1].set_title(f'Recall (μ={mean_recall*100:.1f}%)')
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    for i, class_label in enumerate(sorted_classes):
        axes[1].text(sorted_recalls[i] + 0.01, i, f'{sorted_recalls[i]*100:.0f}', va='center', ha='left')

    # Plot F1 Score
    axes[2].barh([str(class_label) for class_label in sorted_classes], sorted_f1s, color='lightgreen')
    axes[2].set_xlabel('F1 score [%]')
    axes[2].set_title(f'F1 score (μ={mean_f1*100:.1f})')
    axes[2].grid(axis='x', linestyle='--', alpha=0.7)
    for i, class_label in enumerate(sorted_classes):
        axes[2].text(sorted_f1s[i] + 0.01, i, f'{sorted_f1s[i]*100:.0f}%', va='center', ha='left')

    # plt.suptitle('ResNet50: Precision, Recall, and F1 Scores by Class', fontsize=30)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)

    # Save the plot
    output_path = os.path.join(images_root, 'classification_metrics.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Reset font size to default
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

    print(f"[INFO] Saved classification metrics plot to {output_path}")

""" Functions used in inference pipeline """