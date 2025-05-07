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

#sys.path.append("/home/u18/jmculnan/github/tomcat-dataset-creation")

sys.path.append("/Volumes/ssd/00-ckj-publication/conferences/NAACL2025/MultiCAT/sent_emo")

''''
from utils.baseline_utils import (
    update_train_state,
    get_all_batches,
    make_train_state,
    set_cuda_and_seeds,
    plot_train_dev_curve,
    combine_modality_data,
    MultitaskObject,
)
'''

from baseline_utils import (
    update_train_state,
    get_all_batches,
    make_train_state,
    set_cuda_and_seeds,
    plot_train_dev_curve,
    combine_modality_data,
    MultitaskObject,
)

from text_only_model import MultitaskModel, SingleTaskModel
#from baselines.sent_emo.text_only.text_only_model import MultitaskModel, SingleTaskModel


def train_and_predict(
    classifier,
    train_state,
    dataset,
    batch_size,
    num_epochs,
    optimizer,
    tasks,
    device="cpu",
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    """
    num_tasks = len(tasks)
    best_f1 = 0.0

    print(f"Number of tasks: {num_tasks}")
    # get a list of the tasks by number
    for task in range(num_tasks):
        train_state["tasks"].append(task)
        train_state["train_avg_f1"][task] = []
        train_state["val_avg_f1"][task] = []
        train_state["train_macro_f1"][task] = []
        train_state["val_macro_f1"][task] = []
        train_state["val_best_f1"].append(best_f1)

    for epoch_index in range(num_epochs):
        first = datetime.now()
        print(f"Starting epoch {epoch_index} at {first}")

        train_state["epoch_index"] = epoch_index

        # get running loss, holders of ys and predictions on training partition
        running_loss, ys_holder, preds_holder = run_model(
            dataset,
            classifier,
            batch_size,
            tasks,
            device,
            optimizer,
            mode="training",
        )

        # add loss and accuracy to train state
        train_state["train_loss"].append(running_loss)

        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            task_macro_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="macro"
            )
            print(
                f"Training f-scores for task {task}: "
                f"weighted - {task_avg_f1[2]} \t macro - {task_macro_f1[2]}"
            )
            # add training f1 to train state
            train_state["train_avg_f1"][task].append(task_avg_f1[2])
            train_state["train_macro_f1"][task].append(task_macro_f1[2])

        # get running loss, holders of ys and predictions on dev partition
        running_loss, ys_holder, preds_holder = run_model(
            dataset,
            classifier,
            batch_size,
            tasks,
            device,
            optimizer,
            mode="eval",
        )

        all_avg_f1s = []
        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            task_macro_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="macro"
            )

            print(
                f"Val f-scores for task {task}: "
                f"weighted - {task_avg_f1[2]} \t macro - {task_macro_f1[2]}"
            )

            # add val f1 to train state
            train_state["val_avg_f1"][task].append(task_avg_f1[2])
            train_state["val_macro_f1"][task].append(task_macro_f1[2])

            all_avg_f1s.append(task_macro_f1[2])

        # print out classification report if the model will update
        avg_f1_t = sum(all_avg_f1s) / len(all_avg_f1s)

        if avg_f1_t > best_f1:
            for i, item in enumerate(all_avg_f1s):
                train_state["val_best_f1"][i] = item

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
    tasks,
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
    for i in range(len(tasks)):
        ys_holder[i] = []
    preds_holder = {}
    for i in range(len(tasks)):
        preds_holder[i] = []

    # for each batch in the list of batches created by the dataloader
    for batch_index, batch in enumerate(batches):
        # zero gradients
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.zero_grad()

        # get ys and predictions for the batch
        if len(tasks) == 2:
            y_gold = batch["ys"]
        elif tasks[0] == 0 or tasks[0] == "sentiment" or tasks[0] == "sent":
            y_gold = [batch["ys"][0]]
        else:
            y_gold = [batch["ys"][1]]

        batch_pred = predict(
            batch,
            classifier,
            device,
        )
        # add preds to a list if doing single-task prediction
        if len(tasks) != 2:
            batch_pred = [batch_pred]


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
    if type(batch["x_utt"]) == torch.tensor:
        batch_text = batch["x_utt"].detach().to(device)
    else:
        batch_text = batch["x_utt"]

    # feed these parts into classifier
    # compute the output
    batch_pred = classifier(text_input=batch_text)

    return batch_pred


def load_data(config):
    # path for loading data
    load_path = config.load_path

    # set paths to text and audio data
    #text_base = f"{load_path}/MultiCAT/text_data"
    #ys_base = f"{load_path}/MultiCAT/ys_data"

    text_base = f"{load_path}/text_data"
    ys_base = f"{load_path}/ys_data"

    # combine audio, text, and gold label pickle files into a Dataset
    train_text = pickle.load(open(f"{text_base}/train.pickle", "rb"))
    ys_train = pickle.load(open(f"{ys_base}/train.pickle", "rb"))

    print("Combining training text and ys")

    train_data = combine_modality_data([train_text, ys_train])

    del train_text
    del ys_train
    print("Now loading dev data")
    dev_text = pickle.load(open(f"{text_base}/dev.pickle", "rb"))
    ys_dev = pickle.load(open(f"{ys_base}/dev.pickle", "rb"))

    print("Combining dev text and ys")
    dev_data = combine_modality_data([dev_text, ys_dev])

    del dev_text
    del ys_dev

    print("Now loading test data")
    test_text = pickle.load(open(f"{text_base}/test.pickle", "rb"))
    ys_test = pickle.load(open(f"{ys_base}/test.pickle", "rb"))

    print("Combining test text and ys")
    test_data = combine_modality_data([test_text, ys_test])

    # set loss function
    loss_fx = torch.nn.CrossEntropyLoss(reduction="mean")

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
        f"INT-OUTPUT{model_params.fc_hidden_dim}_"
        f"DROPOUT{model_params.dropout}_"
        f"FC-FINALDIM{model_params.final_hidden_dim}",
    )

    # make sure the full save path exists; if not, create it
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(item_output_path))

    # this uses train-dev-test folds
    if model_params.model.lower() == "multitask":
        model = MultitaskModel(
            model_params, device, model_type=config.model_params.model_type
        )
    else:
        model = SingleTaskModel(
            model_params, device, model_type=config.model_params.model_type
        )

    optimizer = torch.optim.Adam(
        lr=model_params.lr,
        params=model.parameters(),
        weight_decay=model_params.weight_decay,
    )

    # set the classifier(s) to the right device
    model = model.to(device)
    print(model)

    # create a save path and file for the model
    model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pt"

    # make the train state to keep track of model training/development
    train_state = make_train_state(
        model_params.lr, model_save_file, model_params.early_stopping_criterion
    )

    # train the model and evaluate on development set
    train_and_predict(
        model,
        train_state,
        dataset,
        model_params.batch_size,
        model_params.num_epochs,
        optimizer,
        ["sentiment", "emotion"] if model_params.model.lower() == "multitask" else [model_params.model.lower()],
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
            train_vals=train_state["train_macro_f1"][item],
            dev_vals=train_state["val_macro_f1"][item],
            y_label="Macro F1",
            title=f"Macro f-scores for task {item} for model {model_params.model} with lr {model_params.lr}",
            save_name=f"{item_output_path}/macro-f1_task-{item}.png",
        )

    # return best sent and emo values
    if model_params.model == "sentiment" or model_params.model == "sent":
        return round(train_state["val_best_f1"][0]* 100, 2), None
    elif model_params.model == "emotion" or model_params.model == "emo":
        return None, round(train_state["val_best_f1"][0]* 100, 2)
    else:
        return round(train_state["val_best_f1"][0]*100, 2), round(train_state["val_best_f1"][1]*100, 2)


if __name__ == "__main__":
    # import parameters for model
    import text_only_config as config
    #import baselines.sent_emo.text_only.text_only_config as config

    device = set_cuda_and_seeds(config)

    print(device)

    print("Now starting data loading")

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

    individual_output = os.path.join(
        output_path,
        f"LR{config.model_params.lr}_BATCH{config.model_params.batch_size}_"
        f"INT-OUTPUT{config.model_params.fc_hidden_dim}_"
        f"DROPOUT{config.model_params.dropout}_"
        f"FC-FINALDIM{config.model_params.final_hidden_dim}",
    )

    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(individual_output))

    # copy the config file into the experiment directory
    shutil.copyfile(config.CONFIG_FILE, os.path.join(individual_output, "config.py"))

    # set holder for sent_f1, emo_f1
    sent_f1 = 0.0
    emo_f1 = 0.0

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            sys.stdout = f

        sent_f1, emo_f1 = finetune(data, device, output_path, config)

    with open(os.path.join(output_path, "all_trials.csv"), "a") as writef:
        writef.write(
            f"{date.today()},{config.model_params.model_type},{config.model_params.model},{config.model_params.lr},"
        )
        writef.write(
            f"{config.model_params.fc_hidden_dim},{config.model_params.final_hidden_dim},{config.model_params.dropout},"
        )
        writef.write(f"{sent_f1},{emo_f1}\n")