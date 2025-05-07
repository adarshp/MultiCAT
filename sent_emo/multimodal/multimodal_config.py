# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

DEBUG = (
    False  # no saving of files; output in the terminal; first random seed from the list
)

# what number experiment is this?
# can leave it at 1 or increment if you have multiple
#   experiments with the same description from the same date
EXPERIMENT_ID = 5
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
EXPERIMENT_DESCRIPTION = "RoBERTaCLS_macrof1_"

# indicate whether this code is being run locally or on the server
USE_SERVER = True

# get this file's path to save a copy
# this does not need to be changed
CONFIG_FILE = os.path.abspath(__file__)

# how many tasks are you running over?
# it's not critical to change this number unless you're running
#   a single dataset over multiple tasks (e.g. asist data)
num_tasks = 2

# where is the preprocessed pickle data saved?
if USE_SERVER:
    load_path = "/xdisk/bethard/jmculnan"
else:
    load_path = "/media/jculnan/One Touch/jculnan/datasets"

# set directory to save full experiments
exp_save_path = "output/multitask"

# set the acoustic and text feature sets
# the first item should be the acoustic feature set
# the second item should be the text embedding type (distilbert, bert, glove)
# the third item is whether to use data in list or dict form
# currently, list form is being phased out, so use dict
# if these items are not set correctly,
# the data may not be loaded properly
feature_set = "IS13_text_dict"

# give a list of the datasets to be used
datasets = ["asist"]

# the number of acoustic features to use
# 130 is the number of features in IS13 set
num_feats = 130

# a namespace object containing the parameters that you might need to alter for training
model_params = Namespace(
    # these parameters are separated into two sections
    # in the first section are parameters that are currently used
    # in the second section are parameters that are not currently used
    #   the latter may either be reincorporated into the network(s)
    #   in the future or may be removed from this Namespace
    # --------------------------------------------------
    # set the random seed; this seed is used by torch and random functions
    seed=88,  # 1007
    # overall model selection
    # --------------------------------------------------
    # 'model' is used to select an overall model during model selection
    # this is in select_model within train_and_test_utils.py
    # options: acoustic_shared, text_shared, duplicate_input, text_only, multitask
    # other options may be added in the future for more model types
    model="Multitask",
    # optimizer parameters
    # --------------------------------------------------
    # learning rate
    # with multiple tasks, these learning rates tend to be pretty low
    # (usually 1e-3 -- 1e-5)
    lr=1e-5,
    # hyperparameters for adam optimizer -- usually not needed to alter these
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=0.0001,
    use_clsswts=True,
    # parameters for model training
    # --------------------------------------------------
    # the maximum number of epochs that a model can run
    num_epochs=100,
    # the minibatch size
    batch_size=8,  # 128,  # 32
    # how many epochs the model will run after updating
    early_stopping_criterion=5,
    # parameters for model architecture
    # --------------------------------------------------
    # number of classes for each of the tasks of interest
    output_0_dim=3,  # number of classes in the sentiment task
    output_1_dim=7,  # number of classes in the emotion task
    # number of layers in the recurrent portion of our model
    # this has actually changed from gru to lstm
    num_gru_layers=2,  # 1,  # 3,  # 1,  # 4, 2,
    # whether the recurrent portion of the model is bidirectional
    bidirectional=True,
    # input dimension parameters
    text_dim=768,  # text vector length # 768 for bert/distilbert, 300 for glove
    short_emb_dim=30,  # length of trainable embeddings vec
    # how long is each audio input -- set by the number of acoustic features above
    audio_dim=num_feats,  # audio vector length
    # hyperparameter for text LSTM
    # the size of the hidden dimension between LSTM layers
    text_gru_hidden_dim=100,  # 30,  # 50,  # 20
    # output dimensions for model
    output_dim=100,  # output dimensions from last layer of base model
    # number of fully connected layers after concatenation of modalities
    # this number must either be 1 or 2
    num_fc_layers=1,  # 1,  # 2,
    # the output dimension of the fully connected layer(s)
    fc_hidden_dim=500,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=300,  # the out size of dset-specific fc1 and input of fc2
    # the dropout applied to layers of the NN model
    # portions of this model have a separate dropout specified
    # it may be beneficial to add multiple dropout parameters here
    # so that each may be tuned
    dropout=0.2,  # 0.2, 0.3
    # parameters that are only used with specific architecture
    # --------------------------------------------------
    # hyperparameters for a text CNN
    # not used unless you are using a CNN text base
    # most of the time, these parameters aren't needed
    kernel_1_size=3,  # first kernel size with 3 convolutional filters
    kernel_2_size=4,  # second kernel size with 3 convolutional filters
    kernel_3_size=5,  # third kernel size with 3 convolutional filters
    out_channels=20,  # number of output channels for text CNN
    text_cnn_hidden_dim=100,  # hidden dimension for text CNN
    # the hidden dimension size for acoustic RNN layers
    # if add_avging is true, this number isn't used,
    #   so it's usually unused
    acoustic_gru_hidden_dim=100,
)
