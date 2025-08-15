
import fastai
from fastai.vision.all import *
import pandas as pd
import os
from pathlib import Path
import torch
import numpy as np
import glob
import time
from datetime import timedelta
import sys

# UNUSED FUNCTION
def merge_csv_files(folder_name, output_file, separator=','):
    dataframe = pd.concat([pd.read_csv(f, sep=separator) for f in glob.glob(f"{folder_name}/*.csv")], ignore_index=True)
    dataframe_tensor = torch.tensor(dataframe.iloc[:, 2:].to_numpy())
    conf, pred_id = torch.max(dataframe_tensor, dim=1)
    dataframe['conf'] = conf
    dataframe['pred_id'] = pred_id
    dataframe.to_csv(output_file, index=False, float_format='%.3f')
    print(f"[INFO] All CSV files merged into {output_file}")

def conduct_plankton_inference(MODEL_NAME, model_weights, TRAIN_DATA_PATH, untarred_dir, CRUISE_NAME, BATCH_SIZE):
    start_time = time.time()
    np.random.seed(42)

    # Set the device to use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device used: {device}")
    print(f'[INFO] FastAI version: {fastai.__version__}')

    # Define the model using FastAI
    # TODO: Rewrite training/inference without FastAI implementation
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=RandomSplitter(),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),
        batch_tfms=[*aug_transforms(
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

    dls = block.dataloaders(TRAIN_DATA_PATH, bs=BATCH_SIZE)
    learn = vision_learner(dls, resnet50, metrics=error_rate)

    # Check for multiple GPUs and use DataParallel if available
    # NOTE: Not debugged; not recommended to use multiple GPU's without coding expertises
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Number of GPU's available: {torch.cuda.device_count()}")
        learn.model = torch.nn.DataParallel(learn.model)

    # Move the model to the appropriate device
    learn.model.to(device)
    learn.load(model_weights, weights_only=False)

    print(f"[INFO] Inference is conducted using the {MODEL_NAME} classifier")
    print(f'[INFO] This is Plankton Identifier version: \t{model_weights}')
    print(f'[INFO] The batch size is set at: \t\t{BATCH_SIZE}')
    print(f'[INFO] The loss function is: \t\t\t{learn.loss_func}')
    print(f"[INFO] Number of categories available: \t\t{len(learn.dls.vocab)}\nWith the following labels: \n{learn.dls.vocab}\n")

    # Define output locations for results and logs
    results_dir = Path(f"data/{CRUISE_NAME}_results")
    results_dir.mkdir(exist_ok=True)
    log_file_path = results_dir / "error_log.txt"

    # In previous versions, iteration was done per date but created large output CSV's (>10GB pre day)
    # Hence, this 'uglier' code iterates over each timestamp
    # os.walk() is not used to prevent loading millions of filepaths into memory
    for date_dir in os.listdir(untarred_dir):
        date_dir_path = os.path.join(untarred_dir, date_dir)
        if not os.path.isdir(date_dir_path): # Structurd like MAIN_FOLDER\DATE\
            continue

        for timestamp in os.listdir(date_dir_path): # Structured like MAIN_FOLDER\DATE\untarred_TIMESTAMP\
            timestamp_path = os.path.join(date_dir_path, timestamp)
            if not (os.path.isdir(timestamp_path)):
                continue

            # Check if the CSV file already exists
            csv_filename = results_dir / f"{CRUISE_NAME}_{date_dir}_{timestamp}.csv"
            if csv_filename.exists():
                print(f"[INFO] CSV file already exists for {csv_filename}, skipping processing.")
                continue
            
            # Retrieve all available images within the folder (including Background.tif files)
            imgs = get_image_files(timestamp_path)
            imgs.sort()

            if len(imgs) == 0:
                print(f"[WARNING] No images found in {timestamp_path}")
                print("=================================================")
                continue

            print(f"[INFO] {len(imgs):,} images in {timestamp_path}")
            #print(f'Path first image: {str(imgs[0])}')
            #print(f'Path last image: {str(imgs[-1])}')

            try:
                dl = learn.dls.test_dl(imgs)
                preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                print("[INFO] Made predictions")

                testdf = pd.DataFrame()
                testdf['id'] = imgs # Image filepaths
                testdf['pred_id'] = label_numeric # Predicted ids

                # Create new column for prediction labels
                label_mapping = {i: class_name for i, class_name in enumerate(learn.dls.vocab)}
                testdf['pred_label'] = testdf['pred_id'].map(label_mapping)

                # Add confidence scores of the other classes
                for class_id in range(len(preds[0])):
                    testdf[class_id] = preds.numpy()[:, class_id]

                # Save individual CSV for each 10-minute bin
                testdf.to_csv(csv_filename, index=False, float_format='%.9f')

            except Exception as e:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"[ERROR] Error processing {timestamp_path}: {e}\n")
                print(f"[ERROR] Error processing {timestamp_path}: {e}")
                sys.exist(1) # Force code to stop


            print(f"[INFO] Finished processing: {timestamp_path}")
            print("=================================================")

    inference_time = time.time()
    elapsed_inference_time = timedelta(seconds=inference_time - start_time).total_seconds()
    print(f"[INFO] Inference completed in {elapsed_inference_time / 3600:.2f} hours.")

    # Merging into single CSV will be phased out in future updates
    # merge_csv_files(results_dir, results_dir / f"{CRUISE_NAME}_all_preds.csv")

    # # For merging all possible CSV's into one large one
    # # This one can cause large memory issues
    # merge_time = time.time()
    # elapsed_merge_time = timedelta(seconds=merge_time - inference_time).total_seconds()
    # print(f"CSV merging completed in {elapsed_merge_time / 3600:.2f} hours.")

    return results_dir