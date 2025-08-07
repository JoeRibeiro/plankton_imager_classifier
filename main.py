from pathlib import Path
import os
import argparse

# Custom modules
from src.untar import extract_tars
from src.remove_corrupted_files import remove_corrupted_files
from src.inference import conduct_plankton_inference
from src.random_samples import get_random_samples
from src.generate_report import create_word_document

# # Hard-coded variables
# MODEL_NAME = "ResNet50-detailed"
# TRAIN_DATA_PATH = Path('data/DETAILED_merged')
# BATCH_SIZE = 600 # Modify this based on your available memory
# CRUISE_NAME = "MONS-Tridens" # For outputs; do not use spaces in this string
# OSPAR = 'data/ospar_comp_au_2023_01_001-gis/ospar_comp_au_2023_01_001.shp' # From: https://odims.ospar.org/en/submissions/ospar_comp_au_2023_01/
# DENSITY_CONSTANT = 340  # This constant is used in the R code for normalization into N per Liter (#/L)

if __name__ == "__main__":
        # Set up argument parser with default values
    parser = argparse.ArgumentParser(description='Plankton Imager Classifier command-line tool.')

    # Mandatory arguments
    parser.add_argument(
        '--source_dir', type=str,
        #default="",
        help='Base directory for source data captured by the Pi-10'
    )

    parser.add_argument(
        '--model_name', type=str,
        #default="ResNet50-detailed",
        help="Please input the name of the model to use. Options are 'OSPAR' and 'ResNet50-Detailed"
    )

    parser.add_argument(
        '--cruise_name', type=str,
        #default="",
        help='Please input the name of the cruise/survey. Used for outputs and intermediate files.'
    )

    # Default arguments
    parser.add_argument(
        '--train_data_path', type=str,
        default="data/DETAILED_merged",
        help='Path to training data. Required for FastAI initialization'
    )

    parser.add_argument(
        '--batch_size', type=int,
        default=600,
        help='Batch size for processing. Adapt based on memory availability.'
    )

    parser.add_argument(
        '--ospar', type=str,
        default='data/ospar_comp_au_2023_01_001-gis/ospar_comp_au_2023_01_001.shp',
        help='Path to OSPAR file'
    )

    parser.add_argument(
        '--density_constant',
        type=int, default=340,
        help='Density constant for normalization to get results in units per liter.'
    )
    args = parser.parse_args()

    # Extract the arguments for use within the individual functions
    SOURCE_BASE_DIR = args.source_dir
    MODEL_NAME = args.model_name
    TRAIN_DATA_PATH = Path(args.train_data_path)
    BATCH_SIZE = args.batch_size
    CRUISE_NAME = args.cruise_name
    OSPAR = args.ospar
    DENSITY_CONSTANT = args.density_constant

    # Define the number of workers to use for parallelization
    max_jobs = min(8, os.cpu_count() or 4)  # Use up to 8 workers or CPU count, whichever is smaller

    # Set the correct model based on user input
    if MODEL_NAME == 'OSPAR':
        # OSPAR classifier for XX number of classes; significantly faster compared to the default option
        print("Not implemented yet...")
        model_weights = ""
        print(f"[INFO] User has chosen to use the {MODEL_NAME} model with weights: {model_weights}")
    else:
        # Default option is the ResNet50 predicting 49 different plankton and non-plankton classes
        model_weights = Path('Plankton_imager_v01_stage-2_Best')
        MODEL_NAME = "ResNet50-Detailed" # Reset the variable in case different spelling is used
        print(f"[INFO] User has chosen to use the {MODEL_NAME} model with weights: {model_weights}")

    # Step 1: Unpack the tarred data
    untarred_dir = extract_tars(SOURCE_BASE_DIR, max_jobs)

    # Step 2: Remove corrupted .tif files
    # No return as modifications are made on the untarred dirs
    remove_corrupted_files(untarred_dir, CRUISE_NAME, max_jobs)

    # Step 3: Conduct inference
    # Note: This is the only script that uses GPU (CPU option available, but discouraged)
    results_dir = conduct_plankton_inference(MODEL_NAME, model_weights, TRAIN_DATA_PATH, untarred_dir, CRUISE_NAME, BATCH_SIZE)
    
    # Step 4: Randomly select n samples of each predicted class for validation and future training iterations
    get_random_samples(results_dir,  CRUISE_NAME, TRAIN_DATA_PATH, model_weights, n_images=100)

    # Step 5: Generate the Word document detailing the cruise
    document_path = create_word_document(results_dir, OSPAR, CRUISE_NAME, DENSITY_CONSTANT, TRAIN_DATA_PATH, model_weights)

    # Step 6: Compress original data for long-term storage
    # Not implemented yet