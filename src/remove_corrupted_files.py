import os
import time
import csv
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, UnidentifiedImageError
from threading import Lock
import hashlib
import tqdm

def calculate_file_hash(file_path, chunk_size=65536):
    """Calculate MD5 hash of a file in chunks to handle large files"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_tiff_corruption(file_path):
    """
    Check if a TIFF file is corrupted.
    Returns a tuple of (is_corrupted, error_message)
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify file integrity

            # img.load()  #<-- Removed as this adds significant I/O overhead

        return (False, None)  # File is not corrupted
    except UnidentifiedImageError:
        return (True, "Not a valid image file")
    except OSError as e:
        return (True, f"File error: {str(e)}")
    except Exception as e:
        return (True, f"Unexpected error: {str(e)}")

def process_file(file_path, corrupted_folder, corrupted_files, lock, cruise_name, file_hashes):
    """
    Process a single file for corruption check and copy if corrupted
    """
    is_corrupted, error_msg = check_tiff_corruption(file_path)

    if is_corrupted:
        # Calculate file hash to check for duplicates
        file_hash = calculate_file_hash(file_path)
        with lock:
            # Check if we've already processed a file with this hash
            if file_hash in file_hashes:
                print(f"[INFO] Skipping duplicate corrupted file: {file_path}")
                # Still record in CSV but with duplicate note
                filename = os.path.basename(file_path)
                dest_path = f"Duplicate of {file_hashes[file_hash]}"
                corrupted_files.append((file_path, f"{error_msg} (Duplicate)", dest_path))
                return True

            # Add to hash set to track duplicates
            file_hashes[file_hash] = file_path

        # Create unique filename in corrupted folder
        filename = os.path.basename(file_path)
        dest_path = os.path.join(corrupted_folder, f"{cruise_name}_{filename}")

        # Ensure unique filename by appending number if needed
        counter = 1
        while os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(corrupted_folder, f"{cruise_name}_{base}_{counter}{ext}")
            counter += 1

        # Acquire lock before shared operations
        with lock:
            corrupted_files.append((file_path, error_msg, dest_path))

            # Copy the file (not move, to preserve original structure)
            try:
                shutil.copy2(file_path, dest_path)
                return True  # Successfully processed corrupted file
            except Exception as e:
                print(f"[WARNING] Error copying {file_path}: {e}")
                return False
    return True

def find_tif_files(directory):
    """
    Recursively find all .tif and .tiff files in a directory
    """
    tif_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                full_path = os.path.join(root, file)
                tif_files.append(full_path)
    return tif_files

def remove_corrupted_files(source_dir, CRUISE_NAME, max_jobs=8, force_reprocess=False):
    """
    Check for corrupted TIFF files in source_dir, copy corrupted files to a new folder,
    and log results to OUTPUT_CSV_PATH. Handles duplicate corrupted files.

    Args:
        source_dir: Directory containing extracted files from untarring process
        CRUISE_NAME: String given by the user indicating the name of the cruise/survey
        max_jobs: Number of parallel workers to use
        force_reprocess: If True, reprocess even if output exists
    """
    start_time = time.time()
    print(f"[INFO] Starting corrupted file detection in: {source_dir}")

    # Construct output path from base folder of untarred data
    base_dir = os.path.dirname(source_dir)
    output_path = os.path.join(base_dir, f"{CRUISE_NAME}_corrupted")
    output_path_csv = os.path.join(output_path, f"{CRUISE_NAME}_corrupted_files.csv")

    # Check if processing has already been done
    if not force_reprocess and os.path.exists(output_path_csv):
        # Check if CSV has content
        if os.path.getsize(output_path_csv) > 0:
            print(f"[INFO] Output CSV file already exists at: {output_path_csv}")
            print("[INFO] This suggests corrupted files have already been processed.")
            print("[INFO] Use force_reprocess=True to reprocess")
            return -1  # Indicate that processing was skipped
    
    # If we're forcing reprocess or CSV doesn't exist, clean up previous output if it exists
    if os.path.exists(output_path):
        print(f"[INFO] Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    if os.path.exists(output_path_csv):
        print(f"[INFO] Removing existing CSV file: {output_path_csv}")
        os.remove(output_path_csv)

    # Create fresh directories
    os.makedirs(output_path, exist_ok=True)

    # Find all TIF files
    print("[INFO] Searching for TIF files...")
    tif_files = find_tif_files(source_dir)
    print(f"[INFO] Found {len(tif_files):,} TIF files to check")

    if not tif_files:
        print("[INFO] No TIF files found. Exiting.")
        return 0  # Return count of corrupted files (0 in this case)

    # Use a manager for thread-safe operations
    lock = Lock()
    corrupted_files = []
    file_hashes = {}  # Dictionary to track file hashes for duplicates

    # Process files in parallel
    print(f"[INFO] Checking files with {max_jobs} workers...")
    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        # Create a wrapper function with all needed parameters
        def process_wrapper(file_path):
            return process_file(file_path, output_path, corrupted_files, lock, CRUISE_NAME, file_hashes)

        # Submit all tasks
        futures = [executor.submit(process_wrapper, file_path) for file_path in tif_files]

        # Wait for all futures to complete and show progress
        for i, future in enumerate(futures):
            result = future.result()
            if (i + 1) % 10000 == 0:  # Print progress every 10,000 files
                print(f"[INFO] Processed {i+1:,}/{len(tif_files):,} files...")

    # Write the corrupted files to the CSV file
    if corrupted_files:
        print(f"[INFO] Found {len(corrupted_files)} corrupted files")
        with open(output_path_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Original File Path", "Error Message", "New Location"])
            for file_path, error_msg, dest_path in corrupted_files:
                writer.writerow([file_path, error_msg, dest_path])

        print(f"[INFO] Corrupted files report saved to: {output_path_csv}")
    else:
        print("[INFO] No corrupted files found!")

    # Calculate and print elapsed time
    elapsed_time = (time.time() - start_time) // 60
    print(f"[INFO] Total execution time: {elapsed_time:,} minutes")

    return len(corrupted_files)
