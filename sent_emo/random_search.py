# implementation of a basic random search
import os
from datetime import date
import shutil
import sys
import random
from copy import deepcopy
import math

import baselines.sent_emo.random_config as config
from utils.baseline_utils import set_cuda_and_seeds


if __name__ == "__main__":
    if config.model_params.modality == "text":
        from baselines.sent_emo.text_only import text_only_train as train
    elif config.model_params.modality == "multimodal":
        from baselines.sent_emo.multimodal import multimodal_train as train
    else:
        exit(
            "This modality is not supported. Please set modality to 'text' or 'multimodal'."
        )

    # get device
    device = set_cuda_and_seeds(config)

    # get a copy of the params
    # this will not be altered during random search
    # the rest of the config file WILL be
    params = deepcopy(config.model_params)

    # load the dataset
    data = train.load_data(config)

    # create save location
    output_path = os.path.join(
        config.exp_save_path,
        str(config.EXPERIMENT_ID)
        + "_"
        + config.EXPERIMENT_DESCRIPTION
        + str(date.today()),
    )

    print(f"OUTPUT PATH:\n{output_path}")

    # make sure the full save path exists; if not, create it
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(output_path))

    # copy the config file into the experiment directory
    shutil.copyfile(config.CONFIG_FILE, os.path.join(output_path, "config.py"))

    # go through the random search to get options
    for i in range(config.num_searches):
        # get deep copy of config file
        this_params = config.model_params

        # get the new config
        this_params.fc_hidden_dim = (
            round(
                random.randint(params.fc_hidden_dim[0], params.fc_hidden_dim[1]) / 50.0
            )
            * 50
        )
        this_params.final_hidden_dim = (
            round(
                random.randint(params.final_hidden_dim[0], params.final_hidden_dim[1])
                / 50.0
            )
            * 50
        )
        this_params.dropout = round(
            random.uniform(params.dropout[0], params.dropout[1]), 1
        )
        this_params.lr = 10 ** math.ceil(
            math.log10(random.uniform(params.lr[0], params.lr[1]))
        )

        # add stdout to a log file
        with open(os.path.join(output_path, "log"), "a") as f:
            if not config.DEBUG:
                sys.stdout = f

            train.finetune(data, device, output_path, config)