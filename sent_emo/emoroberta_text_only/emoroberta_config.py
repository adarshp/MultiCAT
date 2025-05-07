# incorporates elements from https://github.com/marcovzla/discobert/blob/master/config.py

from argparse import Namespace
import os

DEBUG = False  # no saving of files; output in the terminal

# what number experiment is this?
# can leave it at 1 or increment if you have multiple
#   experiments with the same description from the same date
EXPERIMENT_ID = 1
# during training: enter a brief description that will make the experiment easy to identify
# during testing: this is the name of the parent directory for different random seed models saved from an experiment
EXPERIMENT_DESCRIPTION = "EmoRoberta_inference_"

# indicate whether this code is being run locally or on the server
USE_SERVER = False

# get this file's path to save a copy
# this does not need to be changed
CONFIG_FILE = os.path.abspath(__file__)

# how many tasks are you running over?
# it's not critical to change this number unless you're running
#   a single dataset over multiple tasks (e.g. asist data)
num_tasks = 2

# where is the preprocessed pickle data saved?
if USE_SERVER:
    # load_path = "/data/nlp/corpora/MM/pickled_data/distilbert_custom_feats"
    load_path = "/xdisk/bethard/jmculnan"
else:
    load_path = "data"

# set directory to save full experiments
exp_save_path = "output/text_only"

# give a list of the datasets to be used
datasets = ["asist"]

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
    # options: sentiment, emotion, multitask
    # other options may be added in the future for more model types
    model="emotion",
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
    # parameters for model training
    # --------------------------------------------------
    # the maximum number of epochs that a model can run
    num_epochs=50,
    # the minibatch size
    batch_size=8,  # 4  # 128,  # 32
    # how many epochs the model will run after updating
    early_stopping_criterion=5,
    # parameters for model architecture
    # --------------------------------------------------
    # number of classes for each of the tasks of interest
    output_0_dim=3,  # number of classes in the sentiment task
    output_1_dim=7,  # number of classes in the emotion task
    # input dimension parameters
    text_dim=768,  # text vector length # 768 for bert/distilbert, 300 for glove
    short_emb_dim=30,  # length of trainable embeddings vec
    # the output dimension of the fully connected layer(s)
    fc_hidden_dim=768,  # 20,  must match output_dim if final fc layer removed from base model
    final_hidden_dim=192,  # the out size of dset-specific fc1 and input of fc2
    # the dropout applied to layers of the NN model
    # portions of this model have a separate dropout specified
    # it may be beneficial to add multiple dropout parameters here
    # so that each may be tuned
    dropout=0.3,  # 0.2, 0.3
)
