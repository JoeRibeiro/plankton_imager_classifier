import argparse
import tarfile
import tempfile
from pathlib import Path
import pandas as pd
import torch
from fastai.vision.all import *
from src.utils import process_predictions_to_dataframe
from PIL import Image
import os
import shutil
from azureml.core import Dataset, Run
from azureml.core import Workspace, Datastore

def movedatasettoazurecomputeinstance(wsname, wssubscription_id, wsresource_group, dsname):
    ws = Workspace.get(name=wsname, subscription_id=wssubscription_id, resource_group=wsresource_group)
    print(ws)
    dataset = Dataset.get_by_name(ws, name=dsname)
    mount_context = dataset.download(target_path=dsname, overwrite=True)
    return ws, dataset

def get_datastore_path(ws, ds_name):
    datastore = Datastore.get(ws, ds_name)
    return datastore

def upload_file_to_blob(datastore, local_file_path, blob_file_path):
    try:
        print(f"[DEBUG] Uploading {local_file_path} to Azure blob at {blob_file_path}")
        datastore.upload_files(
            files=[local_file_path],
            target_path=blob_file_path,
            overwrite=True,
            show_progress=False
        )
        print(f"[INFO] Successfully uploaded {local_file_path} to {blob_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to upload {local_file_path} to {blob_file_path}: {e}")


# -------------------------
# Create minimal dummy dataset for FastAI
# -------------------------
def ensure_dummy_dataset(train_dataset_path):
    train_dataset_path = Path(train_dataset_path)
    if train_dataset_path.exists() and any(train_dataset_path.iterdir()):
        return  # Already populated

    class_names = ['artefacts','bubbles','copepods','detritus','diatoms','dinoflagellates','fisheggs','fishlarvae','gelatinous','holoplankton','meroplankton','phytoplanktonother'] 
    train_dataset_path.mkdir(parents=True, exist_ok=True)

    for cname in class_names:
        class_path = train_dataset_path / cname
        class_path.mkdir(exist_ok=True)
        for i in range(100):  # Create 100 dummy images per class
            dummy_img = Image.new("L", (10, 10))  # 10x10 grayscale image
            dummy_img.save(class_path / f"dummy_{i}.tif")


# -------------------------
# Process a single .tar file
# -------------------------

def process_single_tar(tar_file_path, model_weights, train_dataset, cruise_name,
                      batch_size=64, density_constant=1.0):
    print(f"[DEBUG] process_single_tar called with tar_file_path: {tar_file_path}")
    print(f"[DEBUG] Checking if path exists: {Path(tar_file_path).exists()}")
    print(f"[DEBUG] Is file: {Path(tar_file_path).is_file()}")
    print(f"[DEBUG] Is dir: {Path(tar_file_path).is_dir()}")
    print(f"[DEBUG] File suffix: {Path(tar_file_path).suffix}")

    ensure_dummy_dataset(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=RandomSplitter(),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    dls = block.dataloaders(train_dataset, bs=batch_size, num_workers=0)
    learn = vision_learner(dls, resnet50, metrics=error_rate, pretrained=False)
    if torch.cuda.device_count() > 1:
        learn.model = torch.nn.DataParallel(learn.model)

    learn.model.to(device)
    abs_model_path = Path(model_weights).resolve()

    target_dir = Path("./models")
    target_dir.mkdir(exist_ok=True)
    target_path = target_dir / Path(model_weights).name
    shutil.copy(model_weights, target_path)

    file_name_no_ext = Path(model_weights).stem
    learn.load(file_name_no_ext, weights_only=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[DEBUG] Temporary directory created at: {temp_dir}")
        try:
            print(f"[DEBUG] Attempting to open tar file: {tar_file_path}")
            with tarfile.open(tar_file_path, "r") as tar:
                print(f"[DEBUG] Tar file opened successfully: {tar_file_path}")
                tar.extractall(path=temp_dir)
                print(f"[DEBUG] Extraction complete to: {temp_dir}")
        except Exception as e:
            print(f"[ERROR] Could not extract {tar_file_path}: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            return None

        imgs = get_image_files(temp_dir)
        print(f"[DEBUG] Number of images found: {len(imgs)}")
        imgs.sort()
        if len(imgs) == 0:
            print(f"[WARNING] No images found in {tar_file_path}")
            return None

        try:
            print(f"[DEBUG] Creating test dataloader for images")
            dl = learn.dls.test_dl(imgs)
            print(f"[DEBUG] Getting predictions")
            preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
            print(f"[DEBUG] Predictions obtained")
        except Exception as e:
            print(f"[WARNING] Prediction failed: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            return None


        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})', tar_file_path)
        if match:
            date_str = match.group(1)
        else:
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        csv_filename, csv_filename_summarized = process_predictions_to_dataframe(
                    imgs=imgs,
                    preds=preds,
                    label_numeric=label_numeric,
                    vocab=learn.dls.vocab,
                    cruise_name=cruise_name,
                    date_str=date_str,
                    time_str=Path(tar_file_path).stem,
                    timestamp_path=temp_dir,
                    results_dir=Path(temp_dir),
                    processed_dir=Path(temp_dir),
                    density_constant=density_constant,
                    csv_filename="inference.csv",
                    tar_file_path=tar_file_path
                )

        print(f"[INFO] Detailed results saved to {csv_filename}")
        print(f"[INFO] Summary results saved to {csv_filename_summarized}")

    return csv_filename



# -------------------------
# CLI interface for Azure ML
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plankton Inference for a Single .tar file")
    parser.add_argument("--tar_file", type=str, required=True)
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--cruise_name", type=str, required=True)
    parser.add_argument("--dsname", type=str, required=True)
    parser.add_argument("--datastorename", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--density_constant", type=float, default=1.0)
    parser.add_argument("--wsname", type=str, default='citdsdp4000nc6s-mlw')
    parser.add_argument("--wssubscription_id", type=str, default='25c7e1b1-ff04-418b-842b-29a58e065da7')
    parser.add_argument("--wsresource_group", type=str, default='CIT-DS-RG-DP4000-NC6S')


    args = parser.parse_args()


    print(args.dsname)
    ws, dataset = movedatasettoazurecomputeinstance(
        wsname=args.wsname,
        wssubscription_id=args.wssubscription_id,
        wsresource_group=args.wsresource_group,
        dsname=args.dsname
    )

    run = Run.get_context()
    ws = run.experiment.workspace
    
    tar_file_path = args.dsname
    

    # Get model
    downloadmodel = Dataset.get_by_name(ws, name=args.model_weights)
    local_model_path = "./model_weights"
    downloadmodel.download(target_path=local_model_path, overwrite=True)

    # Find the .pth file
    import glob
    pth_files = glob.glob(f"{local_model_path}/**/*.pth", recursive=True)
    if not pth_files:
        raise FileNotFoundError("No .pth file found in downloaded dataset")

    resolved_model_weights = pth_files[0]


    print("resolved_model_weights")
    print(resolved_model_weights)



    import glob
    from pathlib import Path

    # After downloading the dataset
    tar_candidate = args.dsname  # This is likely a directory

    # Use glob to find a .tar file inside the directory
    if Path(tar_candidate).is_dir():
        tar_files = glob.glob(f"{tar_candidate}/**/*.tar", recursive=True)
        if tar_files:
            tar_file_path = tar_files[0]
            print(f"[DEBUG] Found tar file inside directory: {tar_file_path}")
        else:
            print(f"[ERROR] No .tar file found inside {tar_candidate}")
            tar_file_path = tar_candidate  # fallback, will error as before
    else:
        tar_file_path = tar_candidate




    csv_result = process_single_tar(
        tar_file_path=tar_file_path,
        model_weights=resolved_model_weights,
        train_dataset=args.train_dataset,
        cruise_name=args.cruise_name,
        batch_size=args.batch_size,
        density_constant=args.density_constant
    )

    if csv_result is not None:
        # Upload to Azure Blob Storage
        datastore = get_datastore_path(ws, args.datastorename)
        blob_file_path = f"{args.dsname}/{os.path.basename(csv_result)}"
        upload_file_to_blob(datastore, csv_result, blob_file_path)
        print(f"Uploaded file to blob: {blob_file_path}")




