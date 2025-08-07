import polars as pl
import os
import shutil

# Custom imports
from src.generate_report import get_pred_labels, clean_df

def get_random_samples(CLASSIFICATION_RESULTS,  CRUISE_NAME, TRAIN_DATA_PATH, MODEL_FILENAME, n_images=100):
    """
    Randomly samples n_images from each class in CLASSIFICATION_RESULTS and organizes them into folders.

    Parameters:
    - CLASSIFICATION_RESULTS: Path to the CSV file containing classification results.
    - CRUISE_NAME: Name of the cruise (unused in this function but kept for compatibility).
    - TRAIN_DATA_PATH: Path to training data (used to get pred_labels).
    - MODEL_FILENAME: Model filename (used to get pred_labels).
    - n_images: Number of images to sample from each class.
    - output_dir: Directory where the class folders will be created.
    """

    # To reduce memory load from ~80GB CSV files, we use Polars + LazyFrames
    # First load in essential information to dynamically loop over the data later on
    lazy_df  = pl.scan_csv(CLASSIFICATION_RESULTS)#, n_rows=200000)
    total_rows = lazy_df.select(pl.len()).collect().item()
    total_classes = lazy_df.select(pl.col("pred_id").unique()).collect().to_series().to_list()
    print(f"[INFO] Read DataFrame. Started processing {total_rows:,} rows.")
    print(f"[INFO] Sampling {n_images} images from each class.")

    # Get actual label names from the model
    pred_labels = get_pred_labels(TRAIN_DATA_PATH, MODEL_FILENAME)

    # Create output directory if it doesn't exist
    output_dir = f"data/{CRUISE_NAME}_sample"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each class to read in less volume, compared ot the entire dataset at once
    # Can still be reasonably high (>10,000,000 rows) with some classes (e.g., detritus)
    for class_id in total_classes:
        print(f"{'-' * 50}")
        print(f"[INFO] Processing class {class_id}")

        # Get subset for current class, excluding Background.tif images
        subset_df = lazy_df.filter(
            (pl.col("pred_id") == class_id) &
            (~pl.col("id").str.contains("Background.tif"))
        ).collect()
        print(f"[INFO] Processing {subset_df.height:,} rows for class {class_id}")

        if subset_df.height == 0:
            print(f"[WARNING] Class {class_id} has no rows. Skipping.")
            continue
        sampled_df = subset_df.sample(n_images, seed=42)
        
        # Sample the prediction label instead of using the numeric label
        pred_label = pred_labels[class_id]

        # Create class directory
        class_dir = os.path.join(output_dir, f"{str(class_id)}_{pred_label}")
        os.makedirs(class_dir, exist_ok=True)
        print(f"[INFO] Created directory for class: {class_dir}")

        # We also save the subset_df for cross-referencing and additional metadata
        excel_path = os.path.join(class_dir, f"{pred_label}_sampled.xlsx")
        cleaned_df, _ = clean_df(sampled_df, pred_labels) # Add proper datetime column to the document 
        cleaned_df.write_excel(excel_path, autofit=True) # Excel instead of CSV for easier use
        
        # Copy files
        filepaths = sampled_df['id'].to_list()
        for src_path in filepaths:
            if not os.path.exists(src_path):
                print(f"[WARNING] File not found: {src_path}. Skipping.")
                continue

            filename = os.path.basename(src_path)
            dest_path = os.path.join(class_dir, filename)

            shutil.copy2(src_path, dest_path)
            print(f"[INFO] Copied {src_path} to {dest_path}")

        print(f"[INFO] Finished processing class {class_id}. Copied {len(sampled_df)} files.")
    print(f"[INFO] Finished processing all classes.")