# prepare pickle files for use in network
# using text-only data
# this ONLY uses the file containing text and gold labels
import pickle
import os
from baselines.sent_emo.data_preprocessing.utils import *


def save_data_components(
    dataset,
    save_path,
    data_path,
    emb_type,
):
    """
    Save partitioned data in pickled format
    :param dataset: the string name of dataset to use
    :param save_path: path where you want to save pickled data
    :param data_path: path to the data
    :param emb_type: whether to use glove or distilbert
    :return:
    """
    dataset = dataset.lower()

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    train_ds, dev_ds, test_ds, clss_weights = prep_data(
        data_path,
        emb_type,
    )
    # save class weights
    pickle.dump(clss_weights, open(f"{save_path}/{dataset}_clsswts.pickle", "wb"))

    all_data = [("train", train_ds), ("dev", dev_ds), ("test", test_ds)]

    for partition_tuple in all_data:
        # get name of partition
        partition_name = partition_tuple[0]
        partition = partition_tuple[1]

        # get utt data + item_ids
        utt_data = get_specific_fields(partition, "utt")
        utt_save_name = f"{save_path}/{dataset}_{emb_type}_{partition_name}"

        # get ys data + item_ids
        ys_data = get_specific_fields(partition, "ys")
        ys_save_name = f"{save_path}/{dataset}_ys_{partition_name}"

        # save
        pickle.dump(utt_data, open(f"{utt_save_name}.pickle", "wb"))
        pickle.dump(ys_data, open(f"{ys_save_name}.pickle", "wb"))


def get_specific_fields(data, field_type, fields=None):
    """
    Partition the data based on a set of keys
    :param data: The dataset
    :param field_type: 'spec', 'acoustic', 'utt', 'ys', or 'other'
    :param fields: if specific fields are given, use this instead of
        field type to get portions of data
    :return: The subset of data with these fields
    """
    sub_data = []
    if fields is not None:
        for item in data:
            sub_data.append(
                {key: value for key, value in item.items() if key in fields}
            )
    else:
        if field_type.lower() == "utt":
            keys = ["x_utt", "utt_length", "audio_id"]
        elif field_type.lower() == "ys":
            keys = ["ys", "audio_id"]
        else:
            exit("Field type not listed, and no specific fields given")

        for item in data:
            sub_data.append({key: value for key, value in item.items() if key in keys})

    return sub_data


def prep_data(
    data_path="../../asist_data2/overall_sent-emo.csv",
    embedding_type="distilbert",
):
    # create instance of StandardPrep class
    asist_prep = SelfSplitPrep(
        data_type="asist",
        data_path=data_path,
        text_type=embedding_type,
    )

    # get train, dev, test data
    train_data, dev_data, test_data = asist_prep.get_data_folds()

    # get train ys
    train_ys = [item["ys"] for item in train_data]

    # get updated class weights using train ys
    class_weights = get_updated_class_weights_multicat(train_ys)

    return train_data, dev_data, test_data, class_weights


if __name__ == "__main__":
    dset = "asist_study4"
    save_path = "output/pickled_data"
    data_path = "~/Downloads/study4_pilot/all_pilot_data_annotated.csv"

    save_data_components(dset, save_path, data_path, emb_type="text")
