import os
import tarfile
import time
from multiprocessing import Pool

def process_tar_file(args):
    """Process a single tar file in a worker process"""
    tar_file, source_base_dir, target_base_dir = args
    try:
        # Calculate relative path to get the date_dir structure
        tar_file_relative = os.path.relpath(tar_file, source_base_dir)
        date_dir = os.path.dirname(tar_file_relative)
        timestamp = os.path.basename(tar_file_relative).replace('.tar', '')

        # Create target directory structure
        target_dir = os.path.join(target_base_dir, date_dir, f"untarred_{timestamp}")
        os.makedirs(target_dir, exist_ok=True)

        # Extract the contents
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(target_dir)

        return (True, tar_file, None)

    except Exception as e:
        error_msg = f"Failed to process {tar_file}: {str(e)}"
        return (False, tar_file, error_msg)

def find_tar_files(source_dir):
    """Find all .tar files in the source directory and subdirectories"""
    # Normalize the path to use forward slashes (helps with path manipulation)
    source_dir = os.path.normpath(source_dir)
    print(f"[DEBUG] Searching in directory: {source_dir}")

    tar_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tar'):
                full_path = os.path.join(root, file)
                tar_files.append(full_path)
    return tar_files

def check_files_extracted(source_dir, target_base_dir):
    """
    Check if all tar files from source directory have already been extracted to target directory.
    Returns True if all files appear to be extracted, False otherwise.
    """
    # Get list of tar files in source directory
    tar_files = find_tar_files(source_dir)
    if not tar_files:
        print("[INFO] No tar files found - nothing to check")
        return True  # Consider this as "nothing to do"

    all_present = True

    for tar_file in tar_files:
        # Get the expected target directory for this tar file
        tar_file_relative = os.path.relpath(tar_file, source_dir)
        date_dir = os.path.dirname(tar_file_relative)
        timestamp = os.path.basename(tar_file_relative).replace('.tar', '')
        target_dir = os.path.join(target_base_dir, date_dir, f"untarred_{timestamp}")

        # Check if the target directory exists and is not empty
        if not os.path.exists(target_dir):
            print(f"[INFO] Target directory not found: {target_dir}")
            all_present = False
            continue

        # Check if the target directory has files
        if not os.listdir(target_dir):
            print(f"[INFO] Target directory is empty: {target_dir}")
            all_present = False
            continue

        # Optional: For a more thorough check, you could compare file contents
        # But for simplicity, we'll just check if the directory exists and isn't empty
        # This is much faster than comparing contents and sufficient for most cases

    return all_present

def extract_tars(source_dir, num_workers=4):
    """
    Main function to extract tar files from the source directory.
    Returns a tuple of (success_count, failure_count, target_dir, elapsed_time)
    """
    # Set up directories
    source_base_dir = source_dir
    target_base_dir = f"{source_base_dir}_UNTARRED"

    # First check if the target directory exists with all files already extracted
    if os.path.exists(target_base_dir):
        print(f"[INFO] Target directory already exists: {target_base_dir}")
        all_present = check_files_extracted(source_base_dir, target_base_dir)

        if all_present:
            print("[INFO] All files appear to be already extracted. Skipping extraction.")
            # Return early with just the target directory info
            return target_base_dir

    # Continue with normal extraction if files aren't already extracted
    # Create target base directory
    os.makedirs(target_base_dir, exist_ok=True)

    print(f"[INFO] Starting extraction from: {os.path.abspath(source_base_dir)}")
    print(f"[INFO] Output directory: {os.path.abspath(target_base_dir)}")

    # Find all tar files
    print("[INFO] Searching for tar files...")
    tar_files = find_tar_files(source_base_dir)
    total_files = len(tar_files)
    print(f"[INFO] Found {total_files:,} tar files to process")

    if not tar_files:
        print("[INFO] No tar files found. Exiting.")
        raise ValueError(f'No tar files were found in {source_dir}. Exiting.')

    print(f"[INFO] Using {num_workers} parallel workers")
    print("[INFO] Starting extraction...")

    # Process files in parallel
    start_time = time.time()

    with Pool(num_workers) as pool:
        # Prepare arguments for each file
        args_list = [(tf, source_base_dir, target_base_dir) for tf in tar_files]
        # Process files and collect results
        results = pool.map(process_tar_file, args_list)

    # Process results
    success_count = 0
    failure_count = 0
    for success, fname, error in results:
        if success:
            success_count += 1
        else:
            failure_count += 1
            print(f"[ERROR] {error}")

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = (end_time - start_time) // 60

    print(f"\n[INFO] Processing complete")
    print(f"[INFO] Successfully processed {success_count} files")
    if failure_count > 0:
        print(f"[WARNING] Failed to process {failure_count} files")
    print(f"[INFO] Total time: {elapsed_time:.2f} minutes")

    return target_base_dir
