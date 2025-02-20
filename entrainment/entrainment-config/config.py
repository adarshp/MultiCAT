# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py
import os
import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
import time
import subprocess
import matplotlib.pyplot as plt
import pdb
import glob
import random
import h5py
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import math
import pprint, pickle
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from scipy import spatial

### ABSOLUTE FILEPATHS FOR INPUT#####

## EDIT THE FOLLOWING LINE TO SET THE DIRECTORY FOR THE FISHER CORPUS
fisher_corpus = "/media/mule/projects/ldc/corpus/" # master directory
##

###############################
# Files in the Fisher Directory
###############################

transcript_dir = fisher_corpus + "/fe_03_p1_tran/data/trans/all_trans" #directory that hourses all transcript files in one directory (no subdirectories)
audio_dir_root = fisher_corpus + "/fisher_eng_tr_sp_LDC2004S13_zip" #directory for sphere sound files
fisher_meta = fisher_corpus + "/Fisher_meta.csv" #metafile, create this before running anything else
#
# 		## Sample files for testing things
def_wav = fisher_corpus + "/" + audio_dir_root + "/fisher_eng_tr_sp_d1/audio/000/fe_03_00004.sph" #example sound file
def_audio = fisher_corpus + "/" + audio_dir_root + "/fisher_eng_tr_sp_d1/audio" #audio subdirectory that houses the sphere file subdirectories
#

###############################
# OUTPUT FILES
###############################

feats_dir = fisher_corpus+"/feats"
data_dir = fisher_corpus+"/feats_nonorm"
raw_feat_dir = fisher_corpus+"/raw_feats"
feats_nonorm_nopre = fisher_corpus+"/feats_nonorm_nopre"

###############################
# MODELLING
###############################

ivec_scp = fisher_corpus + "/Fisher_ivector/exp/ivectors_train/ivector.scp"
model_path = fisher_corpus + "/workspace/acoustic/triplet/fisher/trained_models"
work_dir = fisher_corpus + "/workspace/acoustic/NED_ecdc"
temp_testfile = os.getcwd() + "/models/NED/data/tmp.csv"
fdset = os.getcwd() + "/data/train_Fisher_nonorm.h5"
temp_testfile = os.getcwd() + "/data/tmp.csv"
model_name = os.getcwd() + "/models/trained_VAE_nonorm_nopre_l1.pt"

# do you want to save dataset files?
save_dataset = False

# do you want to load pre-saved dataset files?
load_dataset = True

# get this file's path to save a copy
CONFIG_FILE = os.path.abspath(__file__)

# num_feats = 130
# if feature_set.lower() == "is13":
#     num_feats = 130
# elif "combined_features" in feature_set.lower() or "custom" in feature_set.lower():
#     num_feats = 10

##### modify for model when needed ########
# model_params = Namespace(
#     # use gradnorm for loss normalization
#     use_gradnorm=False,
#     # decide whether to use early, intermediate, or late fusion
#     fusion_type="early",  # int, late, early
#     # consistency parameters
#     seed=88,  # 1007
#     # trying text only model or not
#     text_only=False,
#     audio_only=False,
#     # overall model parameters
#     model="Multitask_text_shared",
#     num_epochs=200,
#     batch_size=100,  # 128,  # 32
#     early_stopping_criterion=5,
#     num_gru_layers=2,  # 1,  # 3,  # 1,  # 4, 2,
#     bidirectional=False,
#     use_distilbert=True,
#     # set whether to have a single loss function
#     single_loss=False,
#     # input dimension parameters
#     text_dim=768,  # text vector length # 768 for bert/distilbert, 300 for glove
#     short_emb_dim=30,  # length of trainable embeddings vec
#     audio_dim=num_feats,  # audio vector length
#     # text NN
#     kernel_1_size=3,
#     kernel_2_size=4,
#     kernel_3_size=5,
#     out_channels=20,
#     text_cnn_hidden_dim=100,
#     # text_output_dim=30,   # 100,   # 50, 300,
#     text_gru_hidden_dim=300,  # 30,  # 50,  # 20
#     # acoustic NN
#     avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
#     add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
#     acoustic_gru_hidden_dim=100,
#     # speaker embeddings
#     use_speaker=False,
#     num_speakers=13,  # check this number
#     speaker_emb_dim=3,
#     # gender embeddings
#     use_gender=False,
#     gender_emb_dim=4,
#     # outputs
#     output_dim=100,  # output dimensions from last layer of base model
#     output_0_dim=2,  # output vec for first task 2 7 5 7 2
#     output_1_dim=7,  # output vec for second task
#     output_2_dim=5,  # output vec for third task
#     output_3_dim=7,
#     output_4_dim=2,
#     # FC layer parameters
#     num_fc_layers=1,  # 1,  # 2,
#     fc_hidden_dim=100,  # 20,  must match output_dim if final fc layer removed from base model
#     final_hidden_dim=50,  # the out size of dset-specific fc1 and input of fc2
#     dropout=0.2,  # 0.2, 0.3
#     # optimizer parameters
#     lr=1e-4,
#     beta_1=0.9,
#     beta_2=0.999,
#     weight_decay=0.0001,
# )
