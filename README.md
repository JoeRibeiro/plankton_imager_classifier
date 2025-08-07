# Plankton Imager Classifier
[[`paper`](google.com)]
[[`dataset`](google.com)]

> The Plankton Imager Classifier predicts 49 different plankton and non-plankton classes from data recovered using the Plankton Imager (Pi-10)[https://www.planktonanalytics.com/]. 

#![Img](./doc/Flowchart_SAMSelect.png)

## Getting Started
### Downloads
* Download both files, and store in /data/
    * Model weights from [[`link`]](google.com)
    * Download the OSPAR data from [[`their website`]](https://odims.ospar.org/en/submissions/ospar_comp_au_2023_01/)

### Anaconda set-up

```
# install the classifier and its dependencies
pip install git+ssh://git@github.com/geoJoost/SAMSelect.git

# Setup the environment
conda create --name mons

conda activate mons

conda install pip

pip install fastai

# IMPORTANT: Modify this installation link to the correct CUDA/CPU version
# See: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

conda install -c conda-forge pandas numpy polars seaborn xlsxwriter
```

### Usage
```
# To start the entire pipeline, navigate to your working directory
cd PATH/TO/WORKING_DIRECTORY

# Run the classifier on demo data
# Not implemented yet
python main.py --source_dir PATH/TO/YOUR/DATA --model_name ResNet50-detailed --cruise_name SURVEY_NAME --batch_size 300

# For more detailed options, see `main.py`
```

## Dataset Requirements
We require the following data to initiate the model:
