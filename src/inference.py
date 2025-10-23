
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
import tarfile, tempfile
from PIL import Image

# Custom modules
from src.remove_corrupted_files import process_corrupted_files
from src.utils import process_predictions_to_dataframe




def ensure_dummy_dataset(model_weights, train_dataset_path):
    # Skip if already populated
    if os.path.exists(train_dataset_path) and len(os.listdir(train_dataset_path)) > 0:
        return
    base = str(model_weights)
    if base.endswith('.pth'):
        base = base[:-4]
    search_paths = [
        os.path.join('models', base + '.pth'),
        base + '.pth'
    ]
    model_file = next((p for p in search_paths if os.path.exists(p)), None)
    if model_file is None:
        raise FileNotFoundError(f"Model weights file not found: tried {search_paths}")
    classnames_csv = model_file + '.classnames.csv'
    if not os.path.exists(classnames_csv):
        classnames_csv = os.path.splitext(model_file)[0] + '.classnames.csv'
    class_names = None
    if os.path.exists(classnames_csv):
        with open(classnames_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            class_names = [row[0] for row in reader if row]
        print(f"[INFO] Loaded {len(class_names)} class names from {classnames_csv}")
    if not class_names:
        state_dict = torch.load(model_file, map_location='cpu')
        num_classes = None
        for key in reversed(state_dict.keys()):
            if key.endswith('.weight') and isinstance(state_dict[key], torch.Tensor):
                weight_shape = state_dict[key].shape
                if len(weight_shape) == 2:
                    num_classes = weight_shape[0]
                    break
        if num_classes is None:
            raise ValueError("Could not determine number of classes from model weights.")
        class_names = [f'class_{i}' for i in range(num_classes)]
        print(f"[INFO] Created {num_classes} generic class folders.")
    os.makedirs(train_dataset_path, exist_ok=True)
    for cname in class_names:
        folder_path = os.path.join(train_dataset_path, cname)
        os.makedirs(folder_path, exist_ok=True)
        for f in os.listdir(folder_path):
            if f.startswith('dummy_') and f.endswith('.tif'):
                os.remove(os.path.join(folder_path, f))
        for i in range(100):
            width = random.randint(10, 100)
            height = random.randint(10, 100)
            img = Image.new('L', (width, height))
            dummy_path = os.path.join(folder_path, f'dummy_{i:03d}.tif')
            img.save(dummy_path)
        print(f"Added 100 dummy images to {folder_path}")
    print(f"[INFO] Created dummy folder structure at '{train_dataset_path}'")
    
    
    
def conduct_plankton_inference(SOURCE_BASE_DIR, MODEL_NAME, model_weights, TRAIN_DATASET, CRUISE_NAME, BATCH_SIZE, DENSITY_CONSTANT, max_jobs):

    print(f"[INFO] Started inference...", flush=True)
    print(f'{SOURCE_BASE_DIR}, {MODEL_NAME}, {model_weights}, {TRAIN_DATASET}, {CRUISE_NAME}, {BATCH_SIZE}, {DENSITY_CONSTANT}, {max_jobs}')
    start_time = time.time()
    np.random.seed(42)
    
    ensure_dummy_dataset(str(model_weights), str(TRAIN_DATASET))
    # Set the device to use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device used: {device}")
    print(f'[INFO] FastAI version: {fastai.__version__}')

    # Define the model using FastAI
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
    dls = block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE, num_workers=0) # Note: on Windows set to 0; can silently fail on HPC systems
    print("DEBUG: FastAI class vocab:", dls.vocab)
    print("DEBUG: Number of classes:", len(dls.vocab))
    print("DEBUG: All image files found:", len(dls.items))
    print("DEBUG: Example image paths and their parent labels:")
    for img_path in dls.items[:20]:  # Show first 20
        print(f"  {img_path} -> {img_path.parent.name}")

    learn = vision_learner(dls, resnet50, metrics=error_rate, pretrained=False)
    
    
    # Check for multiple GPUs and use DataParallel if available
    # NOTE: Not debugged; not recommended to use multiple GPU's without coding expertise
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
    for date_dir in os.listdir(SOURCE_BASE_DIR):
        date_dir_path = os.path.join(SOURCE_BASE_DIR, date_dir)
        if not os.path.isdir(date_dir_path): # See README.md for folder structure
            continue

        for tar_file in os.listdir(date_dir_path):
            if not tar_file.endswith('.tar'):
                continue

            tar_file_path = os.path.join(date_dir_path, tar_file)
            timestamp = tar_file.split('_')[-1].split('.')[0]  # Extract timestamp from tar file name
            date_str = date_dir

            # Check if the last digit is zero (i.e., ends with 0)
            if not timestamp.endswith('0'):  # e.g., "1554" would continue to the next iteration
                print(f"[INFO] Timestamp {timestamp} not a full 10-minute bin, skipping processing.")
                print("=================================================")
                continue

            # Check if the CSV already exists
            csv_filename = results_dir / f"{CRUISE_NAME}_{date_dir}_{timestamp}.csv"
            if csv_filename.exists():
                print(f"[INFO] CSV file already exists for {csv_filename}, skipping processing.")
                continue
            
            # Create temporary location to untar data into
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(tar_file_path, 'r') as tar:
                    try:
                        tar.extractall(path=temp_dir)
                        print(f"[INFO] Extracted {tar_file} to temporary directory")
                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"[ERROR] Could not extract {tar_file}: {e}\n")
                        print(f"[ERROR] Could not extract {tar_file}: {e}")
                        print("=================================================")
                        continue

                    timestamp_path = temp_dir
                    print(f"[DEBUG] timestamp_path: {timestamp_path}")

                    # Retrieve all available images within the folder (including Background.tif files)
                    imgs = get_image_files(timestamp_path)
                    imgs.sort()

                    if len(imgs) == 0:
                        print(f"[WARNING] No images found in {tar_file}")
                        print("=================================================")
                        continue

                    print(f"[INFO] {len(imgs):,} images in {tar_file}")

                    try:
                        try:
                            # First try and make predictions on the given batch
                            dl = learn.dls.test_dl(imgs)
                            preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                            print("[INFO] Made predictions")
                        except:
                            # If corrupted files are found, try to remove these files and re-try the predictions
                            print(f"\n\n[WARNING] Corrupted files found in {timestamp_path}")
                            
                            # Remove corrupted files
                            process_corrupted_files(imgs, timestamp, CRUISE_NAME)

                            # Since we removed several files, we have to reload the available images
                            imgs = get_image_files(timestamp_path)
                            imgs.sort()

                            # Remove any files already marked corrupted
                            corrupt_folder = Path(f"data/{CRUISE_NAME}_corrupted")
                            imgs = [f for f in imgs if not (corrupt_folder / f.name).exists()]

                            # Repeat steps, without corrupted files
                            dl = learn.dls.test_dl(imgs)
                            preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
                            print("[INFO] Made predictions")

                        # Create filename for the detailed CSV
                        csv_filename = results_dir / f"{CRUISE_NAME}_{date_str}_{timestamp}.csv"

                        # Process predictions into dataframes and save to CSV files
                        saved_files = process_predictions_to_dataframe(
                            imgs=imgs,
                            preds=preds,
                            label_numeric=label_numeric,
                            vocab=learn.dls.vocab,
                            cruise_name=CRUISE_NAME,
                            date_str=date_str,
                            time_str=timestamp,
                            timestamp_path=timestamp_path,
                            results_dir=results_dir,
                            processed_dir=processed_dir,
                            density_constant=DENSITY_CONSTANT,
                            csv_filename=csv_filename,
                            tar_file_path=tar_file_path # Path to original .tar file
                        )

                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"[ERROR] Error processing {tar_file}: {e}\n")
                        print(f"[ERROR] Error processing {tar_file}: {e}")
                        sys.exit(1) # Force code to stop

                    print(f"[INFO] Finished processing: {tar_file}")
                    print("=================================================")

    inference_time = time.time()
    elapsed_inference_time = timedelta(seconds=inference_time - start_time).total_seconds()
    print(f"[INFO] Inference completed in {elapsed_inference_time / 3600:.2f} hours.")

    return results_dir, processed_dir