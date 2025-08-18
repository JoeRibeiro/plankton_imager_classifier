""" WIP """

# Plankton Identifier - Classification from plankton imager
# This script provides code to train and run a CNN to classify plankton from images from the plankton imager.
# It treats this problem as a classification task. The labels are extracted from the folder names.

# Import modules and set parameters
import fastai
from fastai.vision.all import *
import torch
import numpy as np
from pathlib import Path
import time

# Custom imports
from src.utils import save_data_visualizations,  save_evaluation_visualizations

def train_resnet50(MODEL_NAME, TRAIN_DATASET, BATCH_SIZE, OUTPUT_NAME):
    np.random.seed(3)

    # Create new folder in /models/ to save .pth files
    # FastAI hard-codes the model part, so have to seperate this for re-use down the line
    models_root = f"{datetime.today().strftime('%Y-%m-%d')}_{MODEL_NAME}" # Use today's date for future note keeping
    os.makedirs(os.path.join('models', models_root), exist_ok=True) # TODO: Prevent overwriting of models

    images_root = os.path.join('doc', models_root)
    os.makedirs(images_root, exist_ok=True)

    # Set the device to use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device used: {device}")
    print(f'[INFO] You are using FastAI version: {fastai.__version__}')

    # Create dataset
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # for regression, change this CategoryBlock
        splitter=RandomSplitter(),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),  # see page 73 book
        batch_tfms=[*aug_transforms(
                mult=1.0,
                do_flip=True,
                flip_vert=True,
                max_rotate=0.2,
                min_zoom=1.0,
                max_zoom=1.1,
                max_lighting=0.3,
                max_warp=0.1,
                p_affine=0.5,
                p_lighting=0.5,
                pad_mode='zeros'
            ),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    dls = block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE, num_workers=0) # num_workers NEEDS to be zero for the code to work on Windows; see https://github.com/fastai/fastai/issues/2899

    # Create various data visualizations for context
    save_data_visualizations(dls, images_root)

    # Create Learner
    learn = vision_learner(dls, resnet50, metrics=error_rate)  # creates pretrained model

    # learn.model.to(device)
    print(f'[INFO] This is Plankton Identifier version: {MODEL_NAME}')
    print(f'[INFO] The batchsize is set at: {BATCH_SIZE}')
    print(f'[INFO] The loss function is: {learn.loss_func}')  # Double check current loss func

    # Save pretrained model
    model_default = os.path.join(models_root, f'{MODEL_NAME}_stage-1_00')
    learn.save(model_default) # Saves pretrained model, for repetitive trials

    # LR finder for frozen model
    learn.lr_find()
    plt.savefig("doc/training/lr_find_frozen.png")

    def train_model(model_file, lr_slice, epochs, save_file, images_root, unfreeze=False):
        """
        Train a model with given parameters.

        Args:
            model_file (str): Path to the model file to load.
            lr_slice (slice): Learning rate slice.
            epochs (int): Number of epochs to train.
            save_file (str): Path to save the trained model.
            unfreeze (bool): Whether to unfreeze the model before training.
        """
        print(f"[INFO] Started training with settings: \nLearning rate: {lr_slice}\nNumber of epochs: {epochs}\nOutput: {save_file}")
        start_time = time.time()

        # Load pre-trained (phase-1) OR best model from phase-1
        learn.load(model_file)

        if unfreeze: # For phase-2
            learn.unfreeze()

        # Perform one cycle learning
        learn.fit_one_cycle(epochs, lr_slice, cbs=SaveModelCallback(monitor='valid_loss', with_opt=True, fname='TempBestModel'))
        
        # Update the current best performing model
        learn.load('TempBestModel')
        learn.save(save_file)

        # Save losses as figure
        learn.recorder.plot_loss() # TODO: Add plt.savefig()
        plt.savefig(os.path.join(images_root, f"{save_file}_losses.png"))

        print(f"[INFO] Model {save_file} training completed in {time.time() - start_time:.2f} seconds")

    # Stage 1 training loops
    print("[INFO] Starting Stage 1 training...\n")
    stage1_params = [
        (slice(9e-3), 1, '_stage1_TEST'),
        # (slice(9e-3), 20, '_stage1_run01'),
        # (slice(9e-2), 20, '_stage1_run02'),
        # (slice(6e-2), 20, '_stage1_run03'),
        # (slice(5e-3), 20, '_stage1_run04'),
        # (slice(10e-3), 20, '_stage1_run05'),
        # (slice(9e-3), 50, '_stage1_run06'),
        # (slice(9e-2), 50, '_stage1_run07'),
        # (slice(6e-2), 50, '_stage1_run08'),
        # (slice(4e-4), 20, '_stage1_run09'),
        # (slice(7e-4), 20, '_stage1_run10'),
    ]
    stage1_models = {} # Save losses to find the most suited model

    for lr_slice, epochs, suffix in stage1_params:
        model_file = MODEL_NAME + suffix
        train_model(model_default, lr_slice, epochs, model_file, images_root, unfreeze=False)

        # Load the model to get its validation loss
        learn.load(model_file)
        val_loss = learn.recorder.val_losses[-1] # Get the last validation loss
        stage1_models[model_file] = val_loss

    # Select the best stage-1 model based on validation loss
    best_stage1_model = min(stage1_models, key=stage1_models.get)
    print(f"[INFO] From stage-1, the best model is: {best_stage1_model}")


    # Define training functions with timing
    def train_stage1(model_file, lr_slice, epochs, save_file):
        print(f"[INFO] Started stage 1 training with settings: \nLearning rate: {lr_slice}\nNumber of epochs: {epochs}]\nOutput: {save_file}")
        start_time = time.time()
        learn.load(model_file)
        learn.fit_one_cycle(epochs, lr_slice, cbs=SaveModelCallback(monitor='valid_loss', with_opt=True, fname='TempBestModel'))
        learn.load('TempBestModel')
        learn.save(save_file)
        learn.recorder.plot_loss()
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def train_stage2(model_file, lr_slice, epochs, save_file):
        start_time = time.time()
        learn.load(model_file)
        learn.unfreeze()
        learn.fit_one_cycle(epochs, lr_slice, cbs=SaveModelCallback(monitor='valid_loss', with_opt=True, fname='TempBestModel'))
        learn.load('TempBestModel')
        learn.save(save_file)
        learn.recorder.plot_loss()
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Stage 1 training loops
    print("Starting Stage 1 training...")
    # First training run
    train_stage1(model_default, slice(9e-3), 20, MODEL_NAME + '_stage-1_01')

    # Additional training runs with different parameters
    train_stage1(model_default, slice(9e-2), 20, models_root + '_stage-1_02')
    train_stage1(model_default, slice(6e-2), 20, models_root + '_stage-1_03')
    train_stage1(model_default, slice(5e-3), 20, models_root + '_stage-1_04')
    train_stage1(model_default, slice(10e-3), 20, models_root + '_stage-1_05')
    train_stage1(model_default, slice(9e-3), 50, models_root + '_stage-1_06')
    train_stage1(model_default, slice(9e-2), 50, models_root + '_stage-1_07')
    train_stage1(model_default, slice(6e-2), 50, models_root + '_stage-1_08')
    train_stage1(model_default, slice(4e-4), 20, models_root + '_stage-1_09')
    train_stage1(model_default, slice(7e-4), 20, models_root + '_stage-1_10')

    # Select best stage-1 model
    learn.load(MODEL_NAME + '_stage-1_02')  # Manually select best stage 1 model
    learn.save(MODEL_NAME + '_stage-1_Best')
    learn.show_results()

    # Stage 2 training loops
    print("Starting Stage 2 training...")
    # First training run
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(1e-6, 1e-4), 20, MODEL_NAME + '_stage-2_01')
    # Additional training runs with different parameters
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(3e-6, 3e-4), 20, MODEL_NAME + '_stage-2_02')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(3e-5, 3e-3), 20, MODEL_NAME + '_stage-2_03')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(3e-7, 3e-5), 20, MODEL_NAME + '_stage-2_04')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(10e-4, 10e-3), 10, MODEL_NAME + '_stage-2_05')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(10e-4, 10e-3), 20, MODEL_NAME + '_stage-2_06')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(10e-4, 10e-3), 10, MODEL_NAME + '_stage-2_07')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(10e-4, 10e-3), 50, MODEL_NAME + '_stage-2_08')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(10e-4, 10e-3), 20, MODEL_NAME + '_stage-2_09')
    train_stage2(MODEL_NAME + '_stage-1_Best', slice(3e-7, 3e-5), 50, MODEL_NAME + '_stage-2_10')

    # Select best stage-2 model
    learn.load(MODEL_NAME + '_stage-2_05')  # select best stage 2 model
    learn.save(MODEL_NAME + '_stage-2_Best')
    learn.show_results()

    # Evaluation
    save_evaluation_visualizations(learn, images_root)