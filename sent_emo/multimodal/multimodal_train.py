# train the models created in models directory with MUStARD data
# currently the main entry point into the system
import shutil
import sys
import os

import torch
from datetime import date, datetime
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from utils.baseline_utils import (
    update_train_state,
    get_all_batches,
    make_train_state,
    set_cuda_and_seeds,
    plot_train_dev_curve,
    combine_modality_data,
    MultitaskObject,
)
from sent_emo.multimodal.multimodal_model import MultitaskModel


def train_and_predict(
    classifier,
    train_state,
    dataset,
    batch_size,
    num_epochs,
    optimizer,
    device="cpu",
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    """
    num_tasks = 2
    best_f1 = 0.0

    print(f"Number of tasks: {num_tasks}")
    # get a list of the tasks by number
    for task in range(num_tasks):
        train_state["tasks"].append(task)
        train_state["train_avg_f1"][task] = []
        train_state["val_avg_f1"][task] = []
        train_state["val_best_f1"].append(0)

    for epoch_index in range(num_epochs):
        first = datetime.now()
        print(f"Starting epoch {epoch_index} at {first}")

        train_state["epoch_index"] = epoch_index

        # get running loss, holders of ys and predictions on training partition
        running_loss, ys_holder, preds_holder = run_model(
            dataset,
            classifier,
            batch_size,
            num_tasks,
            device,
            optimizer,
            mode="training",
        )

        # add loss and accuracy to train state
        train_state["train_loss"].append(running_loss)

        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="macro"
            )
            print(f"Training macro f-score for task {task}: {task_avg_f1}")
            # add training f1 to train state
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # get running loss, holders of ys and predictions on dev partition
        running_loss, ys_holder, preds_holder = run_model(
            dataset,
            classifier,
            batch_size,
            num_tasks,
            device,
            optimizer,
            mode="eval",
        )

        all_avg_f1s = []
        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="macro"
            )
            print(f"Val macro f-score for task {task}: {task_avg_f1}")
            # add val f1 to train state
            train_state["val_avg_f1"][task].append(task_avg_f1[2])
            if task_avg_f1[2] > train_state["val_best_f1"][task]:
                train_state["val_best_f1"][task] = task_avg_f1[2]
            all_avg_f1s.append(task_avg_f1[2])

        # print out classification report if the model will update
        avg_f1_t = sum(all_avg_f1s) / len(all_avg_f1s)

        if avg_f1_t > best_f1:
            best_f1 = avg_f1_t

            for task in preds_holder.keys():
                print(f"Classification report and confusion matrix for task {task}:")
                print(confusion_matrix(ys_holder[task], preds_holder[task]))
                print("======================================================")
                print(
                    classification_report(ys_holder[task], preds_holder[task], digits=4)
                )

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(
            model=classifier, train_state=train_state, optimizer=optimizer
        )

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

        # print out how long this epoch took
        last = datetime.now()
        print(f"Epoch {epoch_index} completed at {last}")
        print(f"This epoch took {last - first}")
        sys.stdout.flush()


def run_model(
    dataset,
    classifier,
    batch_size,
    num_tasks,
    device,
    optimizer,
    mode="training",
):
    """
    Run the model in either training or testing within a single epoch
    Returns running_loss, gold labels, and predictions
    """
    first = datetime.now()

    # Iterate over training dataset
    running_loss = 0.0

    # set classifier(s) to appropriate mode
    if mode.lower() == "training" or mode.lower() == "train":
        classifier.train()
        batches = get_all_batches(dataset, batch_size=batch_size, shuffle=True)
    else:
        classifier.eval()
        batches = get_all_batches(
            dataset, batch_size=batch_size, shuffle=True, partition="dev"
        )

    next_time = datetime.now()
    print(f"Batches organized at {next_time - first}")

    # set holders to use for error analysis
    ys_holder = {}
    for i in range(num_tasks):
        ys_holder[i] = []
    preds_holder = {}
    for i in range(num_tasks):
        preds_holder[i] = []

    # for each batch in the list of batches created by the dataloader
    for batch_index, batch in enumerate(batches):
        # zero gradients
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.zero_grad()

        # get ys and predictions for the batch
        y_gold = batch["ys"]
        batch_pred = predict(
            batch,
            classifier,
            device,
        )

        # calculate loss for each task
        for task, preds in enumerate(batch_pred):
            loss = dataset.loss_fx(preds.to(device), y_gold[task].to(device))
            loss_t = loss.item()

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # use loss to produce gradients
            if mode.lower() == "training" or mode.lower() == "train":
                loss.backward(retain_graph=True)

            # add ys to holder for error analysis
            preds_holder[task].extend(
                [item.index(max(item)) for item in preds.detach().tolist()]
            )
            ys_holder[task].extend(y_gold[task].detach().tolist())

        # increment optimizer
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.step()

    then_time = datetime.now()
    print(f"Train set finished for epoch at {then_time - next_time}")

    return running_loss, ys_holder, preds_holder


def predict(
    batch,
    classifier,
    device,
):
    """
    Get predictions from MultiCAT data
    Used with multitask networks
    """
    # get parts of batches
    # get data
    batch_acoustic = batch["x_acoustic"].detach().to(device)
    if type(batch["x_utt"]) == torch.tensor:
        batch_text = batch["x_utt"].detach().to(device)
    else:
        batch_text = batch["x_utt"]

    batch_lengths = batch["utt_length"]
    batch_acoustic_lengths = batch["acoustic_length"]

    # feed these parts into classifier
    # compute the output
    batch_pred = classifier(
        acoustic_input=batch_acoustic,
        text_input=batch_text,
        length_input=batch_lengths,
        acoustic_len_input=batch_acoustic_lengths,  # todo: change after testing
    )

    return batch_pred


def load_data(config):
    # path for loading data
    load_path = config.load_path

    # set paths to text and audio data
    text_base = f"{load_path}/MultiCAT/text_data"
    audio_base = f"{load_path}/MultiCAT/acoustic_data"
    ys_base = f"{load_path}/MultiCAT/ys_data"
    clsswts_base = f"{load_path}/MultiCAT/class_weights"

    # combine audio, text, and gold label pickle files into a Dataset
    train_text = pickle.load(open(f"{text_base}/train.pickle", "rb"))
    train_audio = pickle.load(open(f"{audio_base}/train.pickle", "rb"))
    ys_train = pickle.load(open(f"{ys_base}/train.pickle", "rb"))

    train_data = combine_modality_data([train_text, train_audio, ys_train])

    dev_text = pickle.load(open(f"{text_base}/dev.pickle", "rb"))
    dev_audio = pickle.load(open(f"{audio_base}/dev.pickle", "rb"))
    ys_dev = pickle.load(open(f"{ys_base}/dev.pickle", "rb"))

    dev_data = combine_modality_data([dev_text, dev_audio, ys_dev])

    test_text = pickle.load(open(f"{text_base}/test.pickle", "rb"))
    test_audio = pickle.load(open(f"{audio_base}/test.pickle", "rb"))
    ys_test = pickle.load(open(f"{ys_base}/test.pickle", "rb"))

    test_data = combine_modality_data([test_text, test_audio, ys_test])

    # get classweights
    clsswts = pickle.load(open(f"{clsswts_base}/asist_clsswts.pickle", "rb"))

    # set loss function
    loss_fx = [
        torch.nn.CrossEntropyLoss(
            reduction="mean",
            weight=clsswts[0].to(device) if config.model_params.use_clsswts else None,
        ),
        torch.nn.CrossEntropyLoss(
            reduction="mean",
            weight=clsswts[1].to(device) if config.model_params.use_clsswts else None,
        ),
    ]

    # convert to dataset
    all_data = MultitaskObject(train_data, dev_data, test_data, loss_fx)

    # return
    return all_data


def finetune(dataset, device, output_path, config):
    model_params = config.model_params

    # 3. CREATE NN
    print(model_params)

    item_output_path = os.path.join(
        output_path,
        f"LR{model_params.lr}_BATCH{model_params.batch_size}_"
        f"NUMLYR{model_params.num_gru_layers}_"
        f"SHORTEMB{model_params.short_emb_dim}_"
        f"INT-OUTPUT{model_params.output_dim}_"
        f"DROPOUT{model_params.dropout}_"
        f"FC-FINALDIM{model_params.final_hidden_dim}",
    )

    # make sure the full save path exists; if not, create it
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(item_output_path))

    # this uses train-dev-test folds
    multitask_model = MultitaskModel(model_params, device)

    optimizer = torch.optim.Adam(
        lr=model_params.lr,
        params=multitask_model.parameters(),
        weight_decay=model_params.weight_decay,
    )

    # set the classifier(s) to the right device
    multitask_model = multitask_model.to(device)
    print(multitask_model)

    # create a save path and file for the model
    model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pt"

    # make the train state to keep track of model training/development
    train_state = make_train_state(
        model_params.lr, model_save_file, model_params.early_stopping_criterion
    )

    # train the model and evaluate on development set
    train_and_predict(
        multitask_model,
        train_state,
        dataset,
        model_params.batch_size,
        model_params.num_epochs,
        optimizer,
        device,
    )

    # plot the loss and accuracy curves
    # set plot titles
    loss_title = f"Training and Dev loss for model {model_params.model} with lr {model_params.lr}"
    loss_save = f"{item_output_path}/loss.png"
    # plot the loss from model
    plot_train_dev_curve(
        train_vals=train_state["train_loss"],
        dev_vals=train_state["val_loss"],
        x_label="Epoch",
        y_label="Loss",
        title=loss_title,
        save_name=loss_save,
    )

    # plot the avg f1 curves for each dataset
    for item in train_state["tasks"]:
        plot_train_dev_curve(
            train_vals=train_state["train_avg_f1"][item],
            dev_vals=train_state["val_avg_f1"][item],
            y_label="Weighted AVG F1",
            title=f"Average f-scores for task {item} for model {model_params.model} with lr {model_params.lr}",
            save_name=f"{item_output_path}/avg-f1_task-{item}.png",
        )

    return train_state["val_best_f1"]


if __name__ == "__main__":
    # import parameters for model
    import baselines.sent_emo.multimodal.multimodal_config as config

    device = set_cuda_and_seeds(config)

    # load the dataset
    data = load_data(config)

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

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            sys.stdout = f

        finetune(data, device, output_path, config)