import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor

def is_corrupted(file_path):
    """Check if a TIFF file is corrupted."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify file integrity
            # img.load()    # Load image data to catch more corruption cases
        return False  # File is not corrupted
    except (UnidentifiedImageError, OSError):
        return True  # File is corrupted
    except Exception as e:
        print(f"Unexpected error checking {file_path}: {e}")
        return False  # Assume not corrupted if there's an unexpected error

def process_file(file_path, timestamp, destination_folder):
    """
    Process a single file - check if it's corrupted and move if needed.
    """

    if is_corrupted(file_path):
        try:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(destination_folder, filename)

            # Handle potential filename conflicts
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(destination_folder, f"{timestamp}_{name}{ext}")
                counter += 1

            shutil.move(file_path, dest_path)
            print(f"Moved corrupted file: {file_path} -> {dest_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def process_corrupted_files(imgs, timestamp, cruise_name, max_jobs=8):
    """
    Check each image in imgs for corruption, and move corrupted images to a destination folder.
    The destination folder will be created as "data/{cruise_name}_corrupted".

    Args:
        imgs: List of file paths to check
        cruise_name: Name to use for the destination folder
        max_jobs: Number of parallel workers to use (default: 8)
    """
    base_dir = "data"
    destination_folder = os.path.join(base_dir, f"{cruise_name}_corrupted")

    # Create the destination folder (and base_dir if needed)
    os.makedirs(destination_folder, exist_ok=True)

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = [executor.submit(process_file, file_path, timestamp, destination_folder)
                  for file_path in imgs]

        # Wait for all tasks to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
