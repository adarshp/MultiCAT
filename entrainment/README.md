------------------------------------------------------------------------------------------
Software supporting "Modeling Vocal Entrainment in Conversational Speech using Deep Unsupervised Learning"
------------------------------------------------------------------------------------------

original code for two-party entrainment (Python2) by Md Nasir 
Modified for Multi-party entrainment (and in Python3)  by Megh Krishnaswamy, Adarsh Pyrelal, and John Culnan.


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

- `Fisher_Model-Training` :- Directory containing scripts for acoustic feature extraction and functional computation, basic code for running training and testing
- `MultiCAT_data_management` :- Directory for working with and testing on all files for multi-party data
- `scripts_and_config`, `praat_scripts` :- Directory containing scripts for setup purposes
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

#### Fisher Data
2. To run the code on your system, download and set-up the LDC data from the kraken server, and access/create `Fisher_meta.csv`.
    - `scp -r [username]@kraken.sista.arizona.edu:/media/mule/projects/ldc [local directory]`
    - Note: You need access.
3. Edit `./Makefile` with relevant filepaths. These are listed in the 'Parameters' section.
4. Run `bash ./Makefile` to extract OpenSMILE features for all sound files.
5. Run ` python feats/feat_extract_nopre.py [all optional arguments]` to extract the feature set. This will require feature files created in the previous step.
    1. Each line contains time- and speaker- and utterance- aligned features for every pair of turns spoken by different speakers.
    2. Note: Do not edit the script, instead, change the optional input parameters while running the script.
6. Run `python feats/create_h5data.py --features_dir [extracted feature directory]`. This will create h5 files with the training, dev and test sets.

#### MultiCAT Data

The transcripts are in `.tsv` format. We will need to remove a colon from the addressee label, sort them in ascending order, and extract and consolidate turns from individual speaker transcripts to create a joint transcript.

1. From the `multicat_data_management` directory, run `multicat_transcript_manage.py`. This script assumes there is a directory   `entrainment_annotations` containing the individual speaker transcripts in `.tsv` format. The output is one `csv` file per trial, saved in `multicat_data_management/files_for_dyad_generation`.
2. Next, we run openSMILE to extract acoustic features. This step assumes the individual speaker audio files are saved in `multicat_data_management/files_for_dyad_generation`, along with the output from the previous step. It will save one `.csv` file per existing dyad per trial to `multicat_data_management/multicat_addressee_feats`
    1. Run `multicat_data_management/generate_dyads_for_addressees.py` to extract acoustic features for turns with addressee labels, split into 3 dyads per trial. 
    2. Run `multicat_data_management/generate_dyads_complete.py` to extract acoustic features for all turns irrespective of addressee labels.
3. Finally generate tensors as an HDF5 file with `multicat_data_management/create_h5_multicat.py`. Now you can test the data.

## Model training
7. Run `python Fisher_Model-Training/train.py`, and provide the right parameters for file and directory paths, and model hyperparameters. You will need the h5 data files for training and validation.
8. The actual model is created using `ecdc.py`: you can make changes to the architecture there.
9. The model will be file with a `.pt` file extension

## Model Testing
10. Run Modify `scripts_and_config` with the right parameters to run `test.py` 30 times. This will write the model output to a text file. This will need the trained model created in the previous step, as well as `test.h5`. This creates the real and fake datasets, and outputs a score for each run.

------------------------
Permissions:
------------------------
ToDo- edit this to reflect the new files
Make sure the following directories/files have permissions:
1. chmod 777 Fisher_Model-Training
2. chmod 777 scripts_and_config/emobase2010_haoqi_revised.conf
