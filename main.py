from pathlib import Path
import os
import argparse
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # For MAC-OS users

# Custom modules
from src.inference import conduct_plankton_inference
from src.random_samples import get_random_samples
from src.generate_report import create_word_document

if __name__ == "__main__":
    print("[INFO] Starting main.py", flush=True)

    # Set up argument parser with default values
    parser = argparse.ArgumentParser(description='Plankton Imager Classifier command-line tool.')

    # Mandatory arguments
    parser.add_argument(
        '--source_dir', type=str,
        default="data/demo_inference",
        help='Base directory for source data captured by the Pi-10'
    )

    parser.add_argument(
        '--model_name', type=str,
        default="OSPAR",
        help="Please input the name of the model to use. Options are 'OSPAR' and 'ResNet50-Detailed"
    )

    parser.add_argument(
        '--cruise_name', type=str,
        default="demo_PIUG",
        help='Please input the name of the cruise/survey. Used for outputs and intermediate files.'
    )

    # Default arguments
    # parser.add_argument(
    #     '--train_data_path', type=str,
    #     default="data/DETAILED_merged",
    #     help='Path to training data. Required for FastAI initialization'
    # )
    

    parser.add_argument(
        '--batch_size', type=int,
        default=128, # Increase for larger machines; 80GB GPU can use batch_size of 600
        help='Batch size for processing. Adapt based on memory availability.'
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
    # TRAIN_DATASET = Path(args.train_data_path)
    BATCH_SIZE = args.batch_size
    CRUISE_NAME = args.cruise_name
    DENSITY_CONSTANT = args.density_constant
    print("[INFO] Arguments received:", args, flush=True)

    # Define the number of workers to use for parallelization
    max_jobs = min(8, os.cpu_count() or 4)  # Use up to 8 workers or CPU count, whichever is smaller

    # Set the correct model based on user input
    if 'ospar' in MODEL_NAME.lower():
        # OSPAR classifier for six number of classes; significantly faster compared to the default option
        model_weights = Path('Plankton_imager_v03_stage-2_Best')
        TRAIN_DATASET = Path('data/OSPAR_merged')
        print(f"[INFO] User has chosen to use the {MODEL_NAME} model with weights: {model_weights}", flush=True)
    else:
        # Default option is the ResNet50 predicting 49 different plankton and non-plankton classes
        model_weights = Path('Plankton_imager_v01_stage-2_Best')
        TRAIN_DATASET = Path('data/DETAILED_merged')
        MODEL_NAME = "ResNet50-Detailed" # Reset the variable in case different spelling is used
        print(f"[INFO] User has chosen to use the {MODEL_NAME} model with weights: {model_weights}, flush=True")
    
    # Check if the model weights file exists
    # FastAI hardcodes the location in /models/ and does not add .pth initially
    if not os.path.exists(os.path.join("models", f"{model_weights}.pth")):
        print(f"Error: The model weights file '{model_weights}' does not exist.", flush=True)
        sys.exit(1)  # Exit with an error code

    # Conduct inference
    # Note: This is the only script that uses GPU (CPU option available, but discouraged)
    results_dir, processed_dir = conduct_plankton_inference(SOURCE_BASE_DIR, # Path to raw .tar dir
                                                            MODEL_NAME, # String pointing to OSPAR or ResNet50 model
                                                            model_weights, # Actual .pth files
                                                            TRAIN_DATASET, # Training dataset required for FastAI
                                                            CRUISE_NAME, # User-defined variable used for outputs
                                                            BATCH_SIZE,  # User-defined variable, how much images to process per batch
                                                            DENSITY_CONSTANT, # 340L per 10 minutes passing through Pi-10
                                                            max_jobs # For parallelization in remove_corrupted_files.py
                                                            )

    # Randomly select n samples of each predicted class for validation and future training iterations
    get_random_samples(results_dir,  CRUISE_NAME, TRAIN_DATASET, model_weights, n_images=100)

    # Generate the Word document detailing the cruise
    document_path = create_word_document(results_dir, CRUISE_NAME, DENSITY_CONSTANT, TRAIN_DATASET, model_weights)

    # Compress original data for long-term storage
    # Not implemented yet