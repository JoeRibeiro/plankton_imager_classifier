import os
import rasterio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def compress_tif_file(img_path, root_dir, output_dir, compression='ZSTD', predictor=2, zlevel=3):
    img_path = os.path.normpath(img_path)
    relative_path = os.path.relpath(os.path.dirname(img_path), root_dir)
    output_subdir = os.path.join(output_dir, relative_path)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, os.path.basename(img_path))
    output_path = os.path.normpath(output_path)

    try:
        with rasterio.open(img_path) as src:
            profile = src.profile.copy()
            profile.update(
                compression=compression,
                predictor=predictor,
                zstd_level=zlevel
            )
            with rasterio.open(output_path, 'w', **profile) as dst:
                for band in range(1, src.count + 1):
                    dst.write(src.read(band), band)
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")

def compress_images(root_dir, output_dir, compression, predictor=2, zlevel=3, max_workers=4):
    files_to_process = []

    # Specifically for the MONS data, we do not want to have long-term storage of the Background.tif files from the PI-10 imager
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.tif') and file != 'Background.tif':
                img_path = os.path.join(subdir, file)
                files_to_process.append(img_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda img_path: compress_tif_file(img_path, 
                                                                  root_dir, 
                                                                  output_dir, 
                                                                  compression,
                                                                  #predictor, 
                                                                  #zlevel
                                                                  ), files_to_process), total=len(files_to_process), desc="Compressing TIFFs"))
        print(f"Outputed saved to: {output_dir}")

if __name__ == "__main__":
    root_directory = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST"
    output_directory = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_COMPRESSED_ZSTD_nopred"
    compress_tif_images(root_directory, output_directory, compression='ZSTD')
