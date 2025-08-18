"""New script for setting up the training of new models"""

# Custom imports
from src.utils import analyze_tif_files, plot_category_examples
from src.train_resnet50 import train_resnet50

def train(MODEL_NAME, TRAIN_DATASET, BATCH_SIZE, OUTPUT_NAME):
    # Plot simple figure describing the training set
    # TODO: Add print statement of dataframe
    # analyze_tif_files(TRAIN_DATASET)

    # Plot examples of each class
    # plot_category_examples(TRAIN_DATASET, output_path='doc/train_example.png')

    # Execute training for ResNet50
    train_resnet50(MODEL_NAME, TRAIN_DATASET, BATCH_SIZE, OUTPUT_NAME)


if __name__ == "__main__":
    MODEL_NAME = "Plankton_imager_TEST"
    TRAIN_DATASET = "data/DETAILED_merged"
    BATCH_SIZE = 32 # 128 for 16G
    OUTPUT_NAME = "TEST_NAME"

    # Execute training regime
    train(MODEL_NAME, TRAIN_DATASET, BATCH_SIZE, OUTPUT_NAME)
    print(f"[INFO] Finished training...")
