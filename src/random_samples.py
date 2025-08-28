import polars as pl
import os
import shutil
from pathlib import Path
import tarfile
import tempfile

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
        print(f'[DEBUG] No CSV files found in {results_dir}')
        raise FileNotFoundError(f"[ERROR] No CSV files found in {results_dir}")
    lazy_df = pl.concat([pl.scan_csv(str(file)) for file in csv_files])
    
    # Check if 'label' column exists and rename only if necessary
    if "label" in lazy_df.collect_schema().keys():
        lazy_df = lazy_df.rename({"label": "pred_id"})
        print("[INFO] Renamed 'label' column to 'pred_id' for backwards compatibility")
    else:
        print("[INFO] No 'label' column found, continuing with 'pred_id'")

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
        os.makedirs(class_dir, exist_ok=True)
        print(f"[INFO] Created directory for class: {class_dir}")

        # We also save the subset_df for cross-referencing and additional metadata
        excel_path = os.path.join(class_dir, f"{pred_label}_sampled.xlsx")
        # cleaned_df, _ = clean_df(sampled_df, pred_labels, class_id) # Add proper datetime column to the document 
        # cleaned_df.write_excel(excel_path, autofit=True) # Excel instead of CSV for easier use
        
        # Re-construct filepaths from .tar datasets
        filenames = sampled_df['id'].str.split("\\").list.last()
        tar_files = sampled_df['tar_file'].to_list()

        # Process each file
        for filename, tar_path in zip(filenames, tar_files):

            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Open tar file and extract the specific file
                with tarfile.open(tar_path, 'r') as tar:
                    # print(tar.getnames())

                    # Clean the internal path
                    path = f"RawImages\\{filename}"

                    # Extract the file
                    tar.extract(path, path=temp_dir)

                    # Construct path to extracted file
                    extracted_path = os.path.join(temp_dir, path)

                    # Construct destination path
                    dest_path = os.path.join(class_dir, os.path.basename(filename))

                    # Move the file
                    shutil.copy2(extracted_path, dest_path)
                    # print(f"[INFO] Extracted and copied {filename} to {dest_path}")

        print(f"[INFO] Finished processing class {class_id}. Copied {len(sampled_df)} files.")
    print(f"[INFO] Finished processing all classes.")