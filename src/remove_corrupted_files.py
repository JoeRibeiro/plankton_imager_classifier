import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError

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

            # shutil.move(file_path, dest_path)
            shutil.copy2(file_path, dest_path)
            print(f"Moved corrupted file: {file_path} -> {dest_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def process_corrupted_files(imgs, timestamp, cruise_name, max_jobs=4, timeout=60):
    """
    Process images in parallel, killing workers if they exceed timeout.
    """
    base_dir = "data"
    destination_folder = os.path.join(base_dir, f"{cruise_name}_corrupted")

    print(f"[INFO] Destination folder: {destination_folder}")
    os.makedirs(destination_folder, exist_ok=True)

    print(f"[INFO] Starting processing with {max_jobs} workers (timeout={timeout}s)")

    with ProcessPoolExecutor(max_workers=max_jobs, mp_context=mp.get_context("spawn")) as executor:
        futures = {
            executor.submit(process_file, f, timestamp, destination_folder): f
            for f in imgs
        }
        print(f"[INFO] All files submitted for processing. Waiting for completion...")

        for i, future in enumerate(futures, 1):
            file_path = futures[future]
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                print(f"[WARNING] Timeout while processing {file_path}")
                # Kill and replace worker
                executor.shutdown(wait=False, cancel_futures=True)
                print("[INFO] Restarting worker pool after timeout...")
                return process_corrupted_files(imgs[i:], timestamp, cruise_name, max_jobs, timeout)
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

            if i % 10000 == 0:
                print(f"[INFO] Processed {i:,}/{len(imgs):,} files")