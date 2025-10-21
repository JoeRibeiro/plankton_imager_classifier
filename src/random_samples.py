import polars as pl
import os
import shutil
from pathlib import Path
import tarfile
import tempfile
import random

# Custom imports
from src.generate_report import get_pred_labels

def get_random_samples(results_dir,  CRUISE_NAME, TRAIN_DATASET, MODEL_FILENAME, n_images=100):
    """
    Randomly samples n_images from each class in results_dir and organizes them into folders.

    Parameters:
    - results_dir: Path to the folder containing CSV files with classification results.
    - CRUISE_NAME: Name of the cruise (unused in this function but kept for compatibility).
    - TRAIN_DATASET: Path to training data (used to get pred_labels).
    - MODEL_FILENAME: Model filename (used to get pred_labels).
    - n_images: Number of images to sample from each class.
    - output_dir: Directory where the class folders will be created.
    """

    # To reduce memory load from ~80GB CSV files, we use Polars + LazyFrames
    # First create glob pattern to find available .csv files
    csv_files = list(Path(results_dir).glob("*.csv"))

    # Get actual label names from the model
    pred_labels = get_pred_labels(TRAIN_DATASET, MODEL_FILENAME)

    if not csv_files:
        print(f'[INFO] No CSV files found in {results_dir}')
        raise FileNotFoundError(f"[ERROR] No CSV files found in {results_dir}")
    
    # As large datasets of Pi-10 (>3TB) can cause memory issues due to CSV's being around 300GB (~3900 files)
    # We limit how many files to process to avoid OOM issues
    MAX_FILES_TO_PROCESS = 1000
    if len(csv_files) > MAX_FILES_TO_PROCESS:
        rng = random.Random(42)
        csv_files = rng.sample(csv_files, MAX_FILES_TO_PROCESS)
        print(f"[INFO] Selected {len(csv_files)} random CSV files out of {len(list(Path(results_dir).glob('*.csv')))}")

    # Process each CSV file individually to ensure consistent schema
    lazy_dfs = []
    for file in list(csv_files):
        # Read the CSV file
        df = pl.scan_csv(str(file))

        if len(df.collect_schema().names()) < len(pred_labels):
            print(f"[WARNING] {file.name} has fewer columns than expected ({len(df.collect_schema().names())}).")
            csv_files.remove(file)
            continue # Skip incomplete CSV's

        # Apply schema corrections immediately
        df = df.with_columns([
            pl.col("density").cast(pl.Float64),
            pl.col("subsample_factor").cast(pl.Float64),
            pl.col("lat").cast(pl.Float64),
            pl.col("lon").cast(pl.Float64),
            pl.col("pred_id").cast(pl.Float64),        # was Int64 / Utf8
            pl.col("pred_conf").cast(pl.Float64),      # was Float64 / Utf8
            pl.col("total_counts").cast(pl.Float64),   # was Float64 / Utf8
        ])

        lazy_dfs.append(df)

    # Concatenate all lazy frames, filling missing columns with nulls
    lazy_df = pl.concat(lazy_dfs, how="diagonal")
        
    total_rows = lazy_df.select(pl.len()).collect().item()
    total_classes = list(range(0, len(pred_labels))) # Get list of one-hot encoded IDs
    print(f"[INFO] Read DataFrame. Started processing {total_rows:,} rows.")
    print(f"[INFO] Sampling {n_images} images from each class.")

    # Create output directory if it doesn't exist
    output_dir = f"data/{CRUISE_NAME}_sample"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each class to read in less volume, compared ot the entire dataset at once
    # Can still be reasonably high (>10,000,000 rows) with some classes (e.g., detritus)
    for class_id in total_classes:
        # Sample the prediction label instead of using the numeric label
        pred_label = pred_labels[class_id]

        print(f"{'-' * 50}")
        print(f"[INFO] Processing class {pred_label}")

        # Get subset for current class, excluding Background.tif images
        subset_df = lazy_df.filter(
            (pl.col("pred_id") == class_id) &
            (~pl.col("id").str.contains("Background.tif"))
        ).collect()
        print(f"[INFO] Processing {subset_df.height:,} rows for class {class_id}")

        if subset_df.height == 0:
            print(f"[WARNING] Class {class_id} has no rows. Skipping.")
            continue
        
        # Randomly sample rows from the DataFrame
        try:
            # Normally you should have more images predicted than num_samples
            sampled_df = subset_df.sample(n=n_images, seed=42)
        except:
            # If you have less, we use with_replacement to handle this case
            # But this should not occur, unless an extremely small dataset is used
            print(f"[WARNING] Only {subset_df.height} images found for class {pred_label} compared to {n_images} required. Sampling {subset_df.height} images instead")
            sampled_df = subset_df.sample(n=subset_df.height, seed=42) # No sample, just take all available images (<80img)

        # Create class directory
        class_dir = os.path.join(output_dir, f"{str(class_id)}_{pred_label}")
        if os.path.exists(class_dir):
            print(f"[INFO] Skipping class {pred_label}. Directory already exists: {class_dir})")
            continue

        os.makedirs(class_dir, exist_ok=True)
        print(f"[INFO] Created directory for class: {class_dir}")
        
        # Re-construct filepaths from .tar datasets
        sampled_df = sampled_df.with_columns([
            pl.col("id")
            .str.replace_all(r"\\", "/")  # Normalize slashes (Windows backslashes)
            .str.split("/")               # Split by slash
            .list.get(-1)                 # Take last part (filename)
            #.str.strip_chars(".tif")      # Remove ".tif"
            .alias("image_id")            # Store as new column
        ])
        filenames = sampled_df['image_id'].to_list()
        tar_files = sampled_df['tar_file'].to_list()

        # Process each file
        for filename, tar_path in zip(filenames, tar_files):
            try:
                # Create temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Open tar file and extract the specific file
                    with tarfile.open(tar_path, 'r') as tar:
                        # Try both slash formats without scanning all entries
                        # Implemented to prevent iterating over all files within the .tar
                        file_obj = None
                        for path in (f"RawImages/{filename}", f"RawImages\\{filename}"):
                            try:
                                file_obj = tar.extractfile(path)
                                if file_obj:
                                    break  # Success
                            except KeyError:
                                continue  # Try next format

                        if file_obj is None:
                            raise FileNotFoundError(
                                f"[ERROR] File '{filename}' not found in tar '{tar_path}'"
                            )

                        # Construct destination path
                        dest_path = os.path.join(class_dir, os.path.basename(filename))

                        # Write to disk directly
                        with open(dest_path, 'wb') as out_f:
                            shutil.copyfileobj(file_obj, out_f)
                            # print(f"[INFO] Extracted and copied {filename} to {dest_path}")
            except Exception as e:
                print(f"[ERROR] Error processing file '{filename}' in tar '{tar_path}': {e}")
                continue
        print(f"[INFO] Finished processing class {class_id}. Copied {len(sampled_df)} files.")
    print(f"[INFO] Finished processing all classes.")