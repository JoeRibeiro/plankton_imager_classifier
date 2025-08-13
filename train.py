"""New script for setting up the training of new models"""


# Custom imports
from src.utils import analyze_tif_files, plot_category_examples
from src.train_resnet50 import train_resnet50

def train(TRAIN_DATA_PATH, BATCH_SIZE, OUTPUT_NAME):
    # Plot simple figure describing the training set
    # TODO: Add print statement of dataframe
    # analyze_tif_files(TRAIN_DATA_PATH)

    # Plot examples of each class
    # plot_category_examples(TRAIN_DATA_PATH, output_path='doc/train_example.png')

    # Execute training
    train_resnet50()


if __name__ == "__main__":

    TRAIN_DATA_PATH = "data/DETAILED_merged"
    BATCH_SIZE = 600
    OUTPUT_NAME = "TEST_NAME"

    # Execute training regime
    train(TRAIN_DATA_PATH, BATCH_SIZE, OUTPUT_NAME)
    print(f"[INFO] Finished training...")
