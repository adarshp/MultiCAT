import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle


class DataIngester:
    def __init__(
        self,
        data_path,
        dataset,
        acoustic_feats="IS13",
        text_feats="distilbert",
        use_spec=False,
    ):
        """
        Get the data and combine as needed
        Assumes the following directory structure
        Data path: contains subdirs for each data type
        {data_path}/acoustic_data/{acoustic_feats} : path to acoustic feats pickle files
        {data_path}/text_data/{text_feats} : path to text feats pickle files
        {data_path}/spectrogram_data : path to spectrogram feats pickle files
        {data_path}/ys_data : path to ys pickle files
        """
        # get dataset
        self.dataset = dataset

        # save acoustic and text feats
        self.a_feats = acoustic_feats
        self.t_feats = text_feats

        # get paths to data
        self.ys_path = f"{data_path}/ys_data"
        self.spec_path = f"{data_path}/spectrogram_data" if use_spec else None
        self.acoustic_path = (
            f"{data_path}/acoustic_data/{acoustic_feats}"
            if acoustic_feats is not None
            else None
        )
        self.text_path = (
            f"{data_path}/text_data/{text_feats}" if text_feats is not None else None
        )

    def get_train_data(self):
        return self._get_data("train")

    def get_dev_data(self):
        return self._get_data("dev")

    def get_test_data(self):
        return self._get_data("test")

    def _get_data(self, partition="train"):
        # load in files
        ys = pickle.load(
            open(f"{self.ys_path}/{self.dataset}_ys_{partition}.pickle", "rb")
        )

        # set other data to None as holder
        spec = None
        acoustic = None
        text = None

        # overwrite if data should be gotten
        if self.spec_path is not None:
            spec = pickle.load(
                open(f"{self.spec_path}/{self.dataset}_spec_{partition}.pickle", "rb")
            )
        if self.acoustic_path is not None:
            acoustic = pickle.load(
                open(
                    f"{self.acoustic_path}/{self.dataset}_{self.a_feats}_{partition}.pickle",
                    "rb",
                )
            )
        if self.text_path is not None:
            text = pickle.load(
                open(
                    f"{self.text_path}/{self.dataset}_{self.t_feats}_{partition}.pickle",
                    "rb",
                )
            )

        data = self._combine_data([ys, spec, acoustic, text])

        return data

    def _combine_data(self, list_of_data):
        """
        Combine data based on what exists
        :param list_of_data : the data (ys, spec, acoustic, text)
            some values may be None
        """
        small_list = [item for item in list_of_data if item is not None]

        combined = pd.DataFrame(small_list[0])
        combined["audio_id"] = combined["audio_id"].astype(str)

        for data in small_list[1:]:
            data = pd.DataFrame(data)
            data["audio_id"] = data["audio_id"].astype(str)
            combined = combined.merge(data, on="audio_id", how="left")

        return combined.to_dict(orient="records")


class MultitaskObject(object):
    """
    An object to hold the data and meta-information for each of the datasets/tasks
    """

    def __init__(
        self,
        train_data,
        dev_data,
        test_data,
        class_loss_func,
    ):
        """
        train_data, dev_data, and test_data are DatumListDataset datasets
        """
        self.train = train_data
        self.dev = dev_data
        self.test = test_data
        self.loss_fx = class_loss_func
        self.loss_multiplier = 1

    def change_loss_multiplier(self, multiplier):
        """
        Add a different loss multiplier to task
        This will be used as a multiplier for loss in multitask network
        e.g. if weight == 1.5, loss = loss * 1.5
        """
        self.loss_multiplier = multiplier


def combine_modality_data(list_of_modality_data):
    """
    Use a list of lists of dicts (each of which contains info on a modality)
    to get a single list of dicts for the dataset
    return this single list of dicts
    """
    all_data = {}

    # get all utterance IDs
    for item in list_of_modality_data[0]:
        all_data[item["audio_id"]] = item

    for dataset in list_of_modality_data[1:]:
        for item in dataset:
            all_data[item["audio_id"]].update(item)

    # return a list of this
    return list(all_data.values())


def get_all_batches(dataset, batch_size, shuffle, partition="train"):
    """
    Create all batches and put them together as a single dataset
    """
    # set holder for batches
    all_batches = []
    all_loss_funcs = []

    if partition == "train":
        data = DataLoader(dataset.train, batch_size=batch_size, shuffle=shuffle)
    elif partition == "dev" or partition == "val":
        data = DataLoader(dataset.dev, batch_size=batch_size, shuffle=shuffle)
    elif partition == "test":
        data = DataLoader(dataset.test, batch_size=batch_size, shuffle=shuffle)
    else:
        sys.exit(f"Error: data partition {partition} not found")

    return data


def make_train_state(learning_rate, model_save_file, early_stopping_criterion):
    # makes a train state to save information on model during training/testing
    return {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 0.0,
        "learning_rate": learning_rate,
        "epoch_index": 0,
        "tasks": [],
        "train_loss": [],
        "train_acc": [],
        "train_avg_f1": {},
        "train_macro_f1": {},
        "val_loss": [],
        "val_acc": [],
        "val_avg_f1": {},
        "val_macro_f1": {},
        "val_best_f1": [],
        "best_val_loss": [],
        "best_val_acc": [],
        "test_avg_f1": {},
        "test_macro_f1": {},
        "best_loss": 100,
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": model_save_file,
        "early_stopping_criterion": early_stopping_criterion,
    }


# plot the training and validation curves
def plot_train_dev_curve(
    train_vals,
    dev_vals,
    x_label="",
    y_label="",
    title="",
    save_name=None,
    show=False,
    axis_boundaries=None,
):
    """
    plot the loss or accuracy/f1 curves over time for training and dev set
    :param train_vals: a list of losses/f1/acc on train set
    :param dev_vals: a list of losses/f1/acc on dev set
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param title: generic plot title; appended with loss or f1
    :param save_name: the name used for saving the image
    :param show: whether to show the image
    :param axis_boundaries: either (lower_bound, upper_bound) tuple
        or None, for automatic boundary setting
    """
    # get a list of the epochs
    epoch = [i for i, item in enumerate(train_vals)]

    # prepare figure
    fig, ax = plt.subplots()
    plt.grid(True)

    # add losses/epoch for train and dev set to plot
    ax.plot(epoch, train_vals, label="train")
    ax.plot(epoch, dev_vals, label="dev")

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # depending on type of input, set the y axis boundaries
    if axis_boundaries:
        ax.set_ylim(axis_boundaries[0], axis_boundaries[1])

    # create title and legend
    ax.set_title(title, loc="center", wrap=True)
    ax.legend()

    # save the file
    if save_name is not None:
        plt.savefig(fname=save_name)
        plt.close()

    # show the plot
    if show:
        plt.show()


def plot_histograms_of_data_classes(
    data_list, x_label="", y_label="", save_name="", show=False
):
    """
    Plot histograms for the number of items per class in the data
    :param data_list: a list containing all gold labels for dataset
    """
    # prepare figure
    fig, ax = plt.subplots()
    plt.grid(True)

    num_classes = set(data_list)

    # add losses/epoch for train and dev set to plot
    ax.hist(data_list, bins=[i for i in range(len(num_classes) + 1)])

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # save the file
    if save_name is not None:
        plt.savefig(fname=save_name)
        plt.close()

    # show the plot
    if show:
        plt.show()


def set_cuda_and_seeds(config):
    # set cuda
    cuda = False
    if torch.cuda.is_available():
        cuda = True

    device = torch.device("cuda" if cuda else "cpu")

    # set random seed
    torch.manual_seed(config.model_params.seed)
    np.random.seed(config.model_params.seed)
    random.seed(config.model_params.seed)
    if cuda:
        torch.cuda.manual_seed_all(config.model_params.seed)

    # check if cuda
    print(cuda)

    return device


def update_train_state(model, train_state, optimizer=None):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state["epoch_index"] == 0:
        if optimizer is not None:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                train_state["model_filename"],
            )
        else:
            torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

        # use val macro f1 instead of val_loss
        macro_f1_t = 0
        for item in train_state["val_macro_f1"].values():
            macro_f1_t += item[-1]
        macro_f1_t = macro_f1_t / len(train_state["tasks"])

        # use best validation accuracy for early stopping
        train_state["early_stopping_best_val"] = macro_f1_t

    # Save model if performance improved
    elif train_state["epoch_index"] >= 1:
        # use val f1 instead of val_loss
        macro_f1_t = 0
        for item in train_state["val_macro_f1"].values():
            macro_f1_t += item[-1]
        macro_f1_t = macro_f1_t / len(train_state["tasks"])

        # if macro f1 is higher
        if macro_f1_t >= train_state["early_stopping_best_val"]:
            # save this as best model
            if optimizer is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    train_state["model_filename"],
                )
            else:
                torch.save(model.state_dict(), train_state["model_filename"])
            print("updating model")
            train_state["early_stopping_best_val"] = macro_f1_t
            train_state["early_stopping_step"] = 0
        else:
            train_state["early_stopping_step"] += 1

        # Stop early ?
        train_state["stop_early"] = (
            train_state["early_stopping_step"]
            >= train_state["early_stopping_criterion"]
        )

    return train_state
