""" WIP """
""" Current code based on Process_02_v01_Inference_on_test_ResNet50.ipynb"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import fastai
from fastai.vision.all import *
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Custom imports
from src.utils import plot_class_distribution, plot_confusion_matrix, plot_classification_metrics

# Hard-coded variables
filename = 'Plankton_imager_v01b' # Insert your filename, for repeated experiments with different trainings hyperparameters (see below)
bs = 128 # Insert highest working batchsize here
ground_truth_file_path = 'data/DETAILED_test.csv'
path = Path('data/DETAILED_merged')
images_path = Path('data/DETAILED_test')

def get_model_predictions(filename, bs, images_path, ground_truth_file_path, path):
    np.random.seed(3)

    # Get image paths
    block = DataBlock(blocks=(ImageBlock, CategoryBlock), # for regression, change this CategoryBlock
                    splitter=RandomSplitter(),
                    get_items = get_image_files, 
                    get_y = parent_label,
                    #item_tfms=Resize((770,1040), method='squish'), 
                    item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'), # see page 73 book
                    batch_tfms=[*aug_transforms( #https://docs.fast.ai/vision.augment#aug_transforms
                        mult=1.0, 
                        do_flip=True,
                        flip_vert=True,
                        max_rotate=0.2,
                        min_zoom=1.0,
                        max_zoom=1.1,
                        max_lighting=0.3,
                        max_warp=0.1,
                        p_affine=0.5,
                        p_lighting=0.5,
                        pad_mode='zeros'), 
                                Normalize.from_stats(*imagenet_stats)] 
                    )

    dls = block.dataloaders(path, bs=bs)


    # Create Learner
    learn = vision_learner(dls, resnet50, #loss_func=Huber(), # different ResNets and Loss functions can be specified here
                        metrics=error_rate
                    ); # creates pretrained model
    learn.model = torch.nn.DataParallel(learn.model) # Parallels computations over multiplle GPUs

    print('This is Plankton Identifier version: ' + filename) # See top
    print('The batchsize is set at:', bs) # See top
    print('The loss function is:', learn.loss_func) # Double check current loss func

    learn.load('Plankton_imager_v01_stage-2_Best', weights_only=False)

    # Path towards model
    #PathModel = Path('data_test');
    

    # Get images to predict
    imgs = get_image_files(images_path);imgs.sort();imgs

    print(f'[INFO] {len(imgs):,} images in {images_path}')
    print(f'The path to the first image is: {imgs[0]}')
    print(f'The path to the last image is: {imgs[-1]}')

    # Create image batch for predicting
    dl = learn.dls.test_dl(imgs)

    # Get predictions for image batch (for large dataset (10k images), this takes ~45min)
    preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
    print(f"[INFO] Made predictions")

    # Create table (i.e. DataFrame) with predictions
    test_df = pd.DataFrame()  # Creates empty table
    test_df['id'] = imgs  # Adds full filepaths
    test_df['filename'] = test_df['id'].apply(lambda x: os.path.basename(x)) # Extract filenames from the full paths in test_df
    test_df['pred_id'] = label_numeric  # Adds predictions to table

    # Create new column for predictions labels
    label_mapping = {i: class_name for i, class_name in enumerate(learn.dls.vocab)}
    test_df['pred_label'] = test_df['pred_id'].map(label_mapping)

    # Load the ground truth CSV file
    ground_truth_df = pd.read_csv(ground_truth_file_path)

    # Rename 'Subdirectory' column to 'Label'
    # TODO: Fix the entire label CSV; either automated from file structure
    ground_truth_df.rename(columns={'Subdirectory': 'label', 'Filename':'filename'}, inplace=True)

    # Merge test_df and ground_truth_df on 'Filename'
    merged_df = pd.merge(test_df, ground_truth_df, on='filename', how='inner')

    # # Define the desired column order
    # # Ensure that all the columns are present in merged_df before reordering
    # desired_order = ['Path', 'Filename', 'Label', 'Pred'] + list(preds_df.columns)

    # # Ensure that all columns in desired_order exist in merged_df
    # existing_columns = [col for col in desired_order if col in merged_df.columns]

    # # Create the Final DataFrame with reordered columns
    # merged_df = merged_df[existing_columns]

    # Export table to .csv file for downstream applications
    csv_path = os.path.join("data", f"{filename}_evaluate.csv")
    merged_df.to_csv(csv_path, index=False, float_format='%.9f')

    # Extracting the true labels and predictions
    y_true = merged_df['label']
    y_pred = merged_df['pred_label']

    # Calculate and display metrics
    correct_predictions = y_pred == y_true
    num_correct_predictions = correct_predictions.sum()

    # Calculating precision, recall and F1 with 'macro' average
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Printing the results
    print(f"Total number of images in test set:\t {len(merged_df):,}")
    print(f"Total number of Correct Predictions:\t {num_correct_predictions:,}")
    print(f"Precision (macro-average):\t\t {precision * 100:.2f}%")
    print(f"Recall (macro-average):\t\t\t {recall * 100:.2f}%")
    print(f"F1 Score (macro-average):\t\t {f1 * 100:.2f}%")

    # Ensure the 'Label' column is sorted alphabetically
    sorted_labels = merged_df['label'].sort_values()

    # Create images root
    images_root = f"doc/{datetime.today().strftime('%Y-%m-%d')}_{filename}_eval"
    os.makedirs(images_root, exist_ok=True)

    # Call visualization functions
    plot_class_distribution(images_root, sorted_labels)
    plot_confusion_matrix(images_root, filename, merged_df, y_true, y_pred)
    plot_classification_metrics(images_root, merged_df)

    # # Plot the histogram
    # plt.figure(figsize=(10, 6))
    # sorted_labels.value_counts().sort_index().plot(kind='bar', color='skyblue')
    # plt.title('Classes in test-set', fontsize=16)
    # plt.xlabel('Label', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.xticks(rotation=90, ha='right', fontsize=8)  # Reduce font size for x-axis labels
    # plt.yticks(fontsize=8)  # Reduce font size for y-axis labels
    # plt.show()

    # # Define your class labels
    # unique_classes = sorted(merged_df['label'].unique())

    # # Compute the confusion matrix
    # cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # # Create a custom annotations array where zeroes are replaced with empty strings
    # annotations = np.where(cm == 0, '', cm)

    # # Create a heatmap to visualize the confusion matrix
    # plt.figure(figsize=(20, 14))

    # sns.heatmap(cm, annot=annotations, fmt='',  # Pass the custom annotations here
    #             cmap='Blues',
    #             cbar=True, 
    #             xticklabels=unique_classes, 
    #             yticklabels=unique_classes,
    #             annot_kws={"size": 10},  # Set the font size for annotations
    #             linewidths=0.5,  # Optional: Add lines between cells for better readability
    #             linecolor='white'  # Optional: Line color between cells
    #         )

    # plt.xlabel('Predicted labels', fontsize=14)
    # plt.ylabel('True labels', fontsize=14)
    # plt.title('Confusion Matrix ResNet50', fontsize=16)
    # plt.xticks(fontsize=12, rotation=-90)
    # plt.yticks(fontsize=12)

    # plt.show()

    # # Font size variable
    # font_size = 18  # Change this value to adjust font size

    # # Set global font size
    # plt.rcParams.update({'font.size': font_size})

    # # Initialize dictionaries for metrics
    # precision_scores = {}
    # recall_scores = {}
    # f1_scores = {}

    # # Calculate metrics
    # for cls in unique_classes:
    #     # Compute precision, recall, and F1 score for each class
    #     precision = precision_score(merged_df['label'], 
    #                                 merged_df['pred_label'], labels=[cls], average='macro', zero_division=0)
    #     recall = recall_score(merged_df['label'], 
    #                         merged_df['pred_label'], labels=[cls], average='macro', zero_division=0)
    #     f1 = f1_score(merged_df['label'], 
    #                 merged_df['pred_label'], labels=[cls], average='macro', zero_division=0)
    #     precision_scores[cls] = precision
    #     recall_scores[cls] = recall
    #     f1_scores[cls] = f1

    # # Convert to lists for plotting
    # sorted_classes = sorted(precision_scores.keys())
    # sorted_precisions = [precision_scores[cls] for cls in sorted_classes]
    # sorted_recalls = [recall_scores[cls] for cls in sorted_classes]
    # sorted_f1s = [f1_scores[cls] for cls in sorted_classes]

    # # Calculate mean values
    # mean_precision = sum(sorted_precisions) / len(sorted_precisions)
    # mean_recall = sum(sorted_recalls) / len(sorted_recalls)
    # mean_f1 = sum(sorted_f1s) / len(sorted_f1s)

    # # Plotting
    # fig, axes = plt.subplots(1, 3, figsize=(30, 20), sharey=True)

    # # Plot Precision
    # axes[0].barh([str(cls) for cls in sorted_classes], sorted_precisions, color='skyblue')
    # axes[0].set_xlabel('Precision [%]')
    # axes[0].set_title(f'Precision for Each Class\nMean Precision: {mean_precision*100:.1f}%')
    # axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    # # Add mean values for each class to the precision panel
    # for i, cls in enumerate(sorted_classes):
    #     axes[0].text(sorted_precisions[i] + 0.01, i, f'{sorted_precisions[i]*100:.0f}', va='center', ha='left')

    # # Plot Recall
    # axes[1].barh([str(cls) for cls in sorted_classes], sorted_recalls, color='salmon')
    # axes[1].set_xlabel('Recall [%]')
    # axes[1].set_title(f'Recall for Each Class\nMean Recall: {mean_recall*100:.1f}%')
    # axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    # # Add mean values for each class to the recall panel
    # for i, cls in enumerate(sorted_classes):
    #     axes[1].text(sorted_recalls[i] + 0.01, i, f'{sorted_recalls[i]*100:.0f}', va='center', ha='left')

    # # Plot F1 Score
    # axes[2].barh([str(cls) for cls in sorted_classes], sorted_f1s, color='lightgreen')
    # axes[2].set_xlabel('F1 Score [%]')
    # axes[2].set_title(f'F1 Score for Each Class\nMean F1 Score: {mean_f1*100:.1f}')
    # axes[2].grid(axis='x', linestyle='--', alpha=0.7)
    # # Add mean values for each class to the F1 score panel
    # for i, cls in enumerate(sorted_classes):
    #     axes[2].text(sorted_f1s[i] + 0.01, i, f'{sorted_f1s[i]*100:.0f}%', va='center', ha='left')

    # plt.suptitle('ResNet50: Precision, Recall, and F1 Scores by Class', fontsize=30)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)  # Adjust the top to make space for suptitle
    # plt.show()

    print(F"[INFO] Finished evaluate.py")

get_model_predictions(filename, bs, images_path, ground_truth_file_path, path)