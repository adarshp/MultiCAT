------------------------------------------------------------------------------------------
Software supporting "Modeling Vocal Entrainment in Conversational Speech using Deep Unsupervised Learning"
------------------------------------------------------------------------------------------

original code by Md Nasir 
Modified for Python3 by Megh Krishnaswamy, Adarsh Pyrelal, and John Culnan.


#### Updated on Feb 15, 2024 (Work in Progress)

#### Scripts are written in Python 3.11


------------
Dependencies and Requirements
------------
1. Bash
2. Python >=3.9 and <=3.11
3. All the Python libraries listed in `setup.py` and env.yaml`. These will be installed 
4. [OpenSMILE](https://github.com/audeering/opensmile)
5. [sph2pipe](https://github.com/burrmill/sph2pipe)
6. [Kaldi](https://kaldi-asr.org)
4. LDC corpus and metadata files <br>(for ToMCAT users, a small sample of the corpus is available at `kraken.sista.arizona.edu:/media/mule/projects/ldc`)

Note: while working with OpenSMILE, remember to add the path to $PATH: `export PATH=$PATH:<path to opensmile dir>/opensmile/build/progsrc/smilextract`

------------------------
Summary of directories and files
--------------------------------

- `feats/` :- Directory containing scripts for acoustic feature extraction and functional computation, basic code for running training and testing
- `Asist3_data_management` :- Directory for working with and testing on all files for multi-party data
- `NED` :- Directory containing code for the NED model
- `scripts_and_config/`, `praat_scripts` :- Directory containing scripts for setup purposes
- `entrainment-config`: location for storing python scripts run commonly across models

------------------------
Files that need to be edited by user to add filepaths:
------------------------

- `./Makefile` :- quick run code

------------------------
Start Point:
------------------------
## Installing Dependencies
1. Setup and activate virtual environment/ conda environment:
    1. Ensure you have access to pip or conda. Installation guide for conda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
    2. Vitual environment: Run `pip install -e .`
    3. Conda environment: Run `conda env create -f env.yaml`

If you wish to extract data, follow the steps in the section [Feature Extraction](#feature-extraction)' below. If you already have the h5 files, skip to the next section.

If you have a trained model, skip to section 'Model training'   

## Feature Extraction
2. To run the code on your system, download and set-up the LDC data from the kraken server, and access/create `Fisher_meta.csv`.
    -   `scp -r [username]@kraken.sista.arizona.edu:/media/mule/projects/ldc [local directory]`
    -   Note: You need access.
3. Edit ./Makefile with relevant filepaths. These are listed in the 'Parameters' section.
4. Run `bash ./Makefile` to extract OpenSMILE features for all sound files.
5. Run ` python feats/feat_extract_nopre.py [all optional arguments]` to extract the feature set. This will require feature files created in the previous step.
    1. Each line contains time- and speaker- and utterance- aligned features for every pair of turns spoken by different speakers.
    2. Note: Do not edit the script, instead, change the optional input parameters while running the script.
6. Run `python feats/create_h5data.py --features_dir [extracted feature directory]`. This will create h5 files with the training, dev and test sets.

## Model training
7. Run `python train.py`, and provide the right parameters for file and directory paths, and model hyperparameters. You will need the h5 data files for training and validation.
8. The actual model is created using `ecdc.py`: you can make changes to the architecture there.
9. The model will be file with a `.pt` file extension

## Model Testing
10. Run `python test.py` 30 times, and write the std output to a file. This will need the trained model created in the previous step, as well as `test.h5`. This creates the real and fake datasets, and outputs a score for each run.
    11. TODO: Create a bash script to do this 

------------------------
Permissions:
------------------------
ToDo- edit this to reflect the new files
Make sure the following directories/files have permissions:
1. chmod 777 feats
2. chmod 777 NED
3. chmod 777 scripts_and_config/emobase2010_haoqi_revised.conf
4. chmod 777 models/NED/emobase2010_revised.conf
