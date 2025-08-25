
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

# Custom modules
from src.remove_corrupted_files import remove_corrupted_files
from src.generate_report import get_geographic_data

def conduct_plankton_inference(MODEL_NAME, model_weights, TRAIN_DATASET, untarred_dir, CRUISE_NAME, BATCH_SIZE, DENSITY_CONSTANT):
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

    dls = block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE)
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
    print(f"[INFO] Number of categories available: \t\t{len(learn.dls.vocab)}")
    print(f"[INFO] Available labels: {learn.dls.vocab}")

    # Define output locations for results and logs
    results_root = Path(f"data/{CRUISE_NAME}_results")
    results_dir = results_root / "raw"
    processed_dir = results_root / "processed"

    # Create all directories (parents=True ensures parent directories are created if needed)
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Location for logs
    log_file_path = results_root / "error_log.txt"

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

            # Extract timestamp like '1550'
            time_str = re.search(r'(\d{4})', timestamp).group(1)

            # Check if the last digit is zero (i.e., ends with 0)
            if not time_str.endswith('0'):  # e.g., "1554" would continue to the next iteration
                print(f"[INFO] Timestamp {time_str} doesn't end with 0, skipping processing.")
                #print("=================================================")
                continue

            # Check if the CSV already exists
            csv_filename = results_dir / f"{CRUISE_NAME}_{date_dir}_{time_str}.csv"
            if csv_filename.exists():
                print(f"[INFO] CSV file already exists for {csv_filename}, skipping processing.")
                continue
            
            # Retrieve all available images within the folder (including Background.tif files)
            imgs = get_image_files(timestamp_path)
            imgs.sort()

            imgs = imgs[:100]

            if len(imgs) == 0:
                print(f"[WARNING] No images found in {timestamp_path}")
                print("=================================================")
                continue

            print(f"[INFO] {len(imgs):,} images in {timestamp_path}")

            try:
                try:
                    # First try and make predictions on the given batch
                    dl = learn.dls.test_dl(imgs)
                    preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                    print("[INFO] Made predictions")
                except:
                    # If corrupted files are found, try to remove these files and re-try the predictions
                    print(f"[WARNING] Corrupted files found in {timestamp_path}")
                    
                    # Remove corrupted files
                    remove_corrupted_files(timestamp_path, CRUISE_NAME, max_jobs=8) # TODO: Replace with variable
                    
                    # Repeat steps, without corrupted files
                    dl = learn.dls.test_dl(imgs)
                    preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                    print("[INFO] Made predictions")

                # Make DataFrame of outputs
                testdf = pd.DataFrame()
                testdf['id'] = imgs # Image filepaths
                testdf['cruise_name'] = CRUISE_NAME

                # Date/time column
                testdf['date'] = pd.to_datetime(date_dir) # Like '2042-06-24'
                testdf['time'] = pd.to_datetime(time_str, format='%H%M').strftime('%H:%M') # Looks for 4 digits like '1530'
                testdf['datetime'] = pd.to_datetime(testdf['date'].astype(str) + ' ' + testdf['time'].astype(str))
                
                # Create new column for the pi-sensor used. Code split in multiple rows for better readability
                filtered_df = testdf[~testdf['id'].astype(str).str.contains("Background.tif")] # Guarantee we are not looking at Background.tif rows
                path_str = str(filtered_df['id'].iloc[0])# Get the first 'id' column
                path_parts = re.split(r'[/\\]', path_str) # Split the path by both '/' and '\' to handle mixed separators
                testdf['instrument_code'] = path_parts[-1].split('.')[0] # Get the pi_code

                # Create new column for prediction labels
                label_mapping = {i: class_name for i, class_name in enumerate(learn.dls.vocab)} # Assign data like: class_id:class_label
                testdf['pred_id'] = label_numeric # Predicted ids
                testdf['pred_label'] = testdf['pred_id'].map(label_mapping) # Predicted labels
                testdf['pred_conf'] = [preds.numpy()[i, pred_id] for i, pred_id in enumerate(testdf['pred_id'])] # Prediction confidence

                # Add confidence scores of the other classes with named columns
                for class_id, class_name in label_mapping.items():
                    testdf[f"{class_name}_conf"] = preds.numpy()[:, class_id]

                # Create a secondary dataframe based on the first with summary statistics
                def summarize_predictions(testdf, timestamp_path, DENSITY_CONSTANT):
                    # Split data into background and non-background
                    # Background is used for generating lat-lon later on
                    filtered_df = testdf[~testdf['id'].astype(str).str.contains("Background.tif")].copy()

                    # Define column groups
                    metadata_cols = ['cruise_name', 'date', 'time', 'datetime', 'instrument_code']
                    conf_columns = [col for col in testdf.columns if col.endswith('_conf')]
                    numeric_cols = ['pred_id', 'pred_conf']  # Add other numeric columns as needed
                    grouping_col = 'pred_label'

                    metadata = filtered_df.iloc[0][metadata_cols].to_dict() # Get duplicate/static values
                    all_categories = filtered_df[grouping_col].unique().tolist() # Get possible categories from the filtered data

                    # Create summary statistics from filtered data
                    grouped = filtered_df.groupby(grouping_col).agg({
                        **{col: 'mean' for col in conf_columns + numeric_cols},
                        grouping_col: 'count'
                    }).rename(columns={grouping_col: 'total_counts'})

                    # Reset index and merge with all categories
                    summary_df = (
                        pd.DataFrame({grouping_col: all_categories})
                        .merge(grouped, on=grouping_col, how='left')
                        .fillna(0)  # Fill missing values with 0
                    )

                    # Add metadata columns with their first values
                    for col in metadata_cols:
                        summary_df[col] = metadata.get(col, None)

                    # Create new column for density. NOTE: This is currently static value for 10-minute bin
                    with open(os.path.join(timestamp_path, "HitsMisses.txt")) as f:
                        lines = f.read().strip().split("\n") # Remove any leading/trailing whitespace

                        # Process each line to extract the numbers
                        # First column: number of ROIs that have been detected AND which are saved onto disk
                        # Second column: number of ROIs that have been detected which are NOT saved onto disk
                        hits, misses = [], []
                        for line in lines:
                            parts = line.split(',')
                            hit, miss = map(int, parts)  # Convert to integers
                            hits.append(hit), misses.append(miss)
                        
                        # Calculate subsample factor to account for missing pictures
                        total_hits = sum(hits)
                        total_misses = sum(misses)
                        summary_df['subsample_factor'] = total_hits / (total_hits + total_misses)
                        print(f"[INFO] Hits: {total_hits:,} | Misses: {total_misses:,}")

                        # Volume = DENSITY_CONSTANT (set to 340L per 10 minutes) 
                        summary_df['density'] = (summary_df['total_counts'] / summary_df['subsample_factor']) / DENSITY_CONSTANT
                    
                    # Add geographic data to output
                    background_df = testdf[testdf['id'].astype(str).str.contains("Background.tif")].copy()
                    lat, lon = get_geographic_data(background_df['id'].iloc[0]) # Using the filepath, we can extract EXIF data
                    summary_df[['lat', 'lon']] = lat, lon

                    # Define column order
                    column_order = (
                        metadata_cols +
                        [grouping_col, 'pred_id', 'pred_conf', 'total_counts', 'density', 'subsample_factor', 'lat', 'lon'] + 
                        sorted([col for col in conf_columns if col in summary_df.columns]) # Misc. confidence columns
                    )

                    # Apply column ordering
                    columns_order = [col for col in column_order if col in summary_df.columns]
                    summary_df = summary_df[columns_order].sort_values(by='pred_label')

                    # Explicitly cast total_counts to float64 in pandas
                    summary_df['total_counts'] = summary_df['total_counts'].astype('float64')

                    return summary_df, columns_order
                
                # Generate a summarized version of the report for each 10-minute bin
                summary_df, columns_order  = summarize_predictions(testdf, timestamp_path, DENSITY_CONSTANT)
                csv_filename_summarized = processed_dir / f"{CRUISE_NAME}_{date_dir}_{time_str}_summary.csv"
                summary_df.to_csv(csv_filename_summarized, index=False, float_format='%.2f') # Use two decimals since this is meant to be human readable

                # Remove Background.tif rows
                testdf = testdf[~testdf['id'].astype(str).str.contains("Background.tif")].copy()


                # Take some of the generated columns and map back into longer .csv
                # Merge with testdf on 'pred_label' to bring over all needed columns
                testdf = testdf.merge(
                    summary_df[['pred_label', 'lat', 'lon', 'subsample_factor', 'total_counts', 'density']],
                    on='pred_label',
                    how='left'
                )

                # Re-arrange columns for extensive .csv based on the previous output
                testdf = testdf[['id'] + columns_order]

                # Save individual CSV for each 10-minute bin
                testdf.to_csv(csv_filename, index=False, float_format='%.9f')

            except Exception as e:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"[ERROR] Error processing {timestamp_path}: {e}\n")
                print(f"[ERROR] Error processing {timestamp_path}: {e}")
                sys.exit(1) # Force code to stop

            print(f"[INFO] Finished processing: {timestamp_path}")
            print("=================================================")

    inference_time = time.time()
    elapsed_inference_time = timedelta(seconds=inference_time - start_time).total_seconds()
    print(f"[INFO] Inference completed in {elapsed_inference_time / 3600:.2f} hours.")

    return results_dir, processed_dir