""""

NOT IMPLEMENTED!! WORK IN PROGRESS



"""



import os
import subprocess
import glob
import tarfile
import argparse

def compress_individually(directory):
    """Compress each .tar file in the directory individually using xz."""
    tar_files = glob.glob(os.path.join(directory, '*.tar'))

    for tar_file in tar_files:
        print(f"Compressing {tar_file}...")
        subprocess.run(['xz', '-z', tar_file], check=True)
        print(f"Created {tar_file}.xz")

def combine_and_compress(directory, output_name='combined'):
    """Combine all .tar files in the directory into one and compress with xz."""
    tar_files = glob.glob(os.path.join(directory, '*.tar'))

    if not tar_files:
        print("No .tar files found in the directory.")
        return

    combined_tar = os.path.join(directory, f'{output_name}.tar')
    with tarfile.open(combined_tar, 'w') as tar:
        for tar_file in tar_files:
            tar.add(tar_file, arcname=os.path.basename(tar_file))
            print(f"Added {tar_file} to {combined_tar}")

    print(f"Compressing {combined_tar}...")
    subprocess.run(['xz', '-z', combined_tar], check=True)
    print(f"Created {combined_tar}.xz")

    os.remove(combined_tar)
    print(f"Removed intermediate file {combined_tar}")

def main():
    parser = argparse.ArgumentParser(description='Compress .tar files for long-term storage.')
    parser.add_argument('directory', help='Directory containing .tar files')
    parser.add_argument('--no-combine', action='store_true', help='Compress each .tar file individually instead of combining')
    parser.add_argument('--output', default='combined', help='Output filename prefix when combining (default: combined)')

    args = parser.parse_args()

    if args.no_combine:
        compress_individually(args.directory)
    else:
        combine_and_compress(args.directory, args.output)

if __name__ == '__main__':
    main()
