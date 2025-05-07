import torch
from sklearn.utils import compute_class_weight
import pandas as pd
from sent_emo.data_preprocessing.text_tokenization_utils import *
from torchtext.data import get_tokenizer
from tqdm import tqdm
import datetime
import random
import math


def get_updated_class_weights_multicat(train_ys):
    """
    Get updated class weights
    Because DataPrep assumes you only enter train set
    :return:
    """
    all_task_weights = []
    # get class weights for each task
    for i in range(len(train_ys[0])):
        labels = [int(y[i]) for y in train_ys]
        classes = sorted(list(set(labels)))
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights = torch.tensor(weights, dtype=torch.float)
        all_task_weights.append(weights)

    return all_task_weights


class SelfSplitPrep:
    """
    A class for when data must be manually partitioned
    """

    def __init__(
        self,
        data_type,
        data_path,
        train_prop=0.6,
        test_prop=0.2,
        text_type="distilbert",
    ):
        # set path to data files
        self.d_type = data_type.lower()

        # set train and test proportions
        self.train_prop = train_prop
        self.test_prop = test_prop

        # read in data
        self.all_data = pd.read_csv(data_path)

        # get dict of all speakers to use
        all_speakers = set(self.all_data["participant"])
        self.speaker2idx = get_speaker_to_index_dict(all_speakers)
        # drop na's for sent-emo labels
        self.all_data = self.all_data.dropna(subset=["sentiment", "emotion"])

        # set tokenizer
        if text_type.lower() == "bert":
            self.tokenizer = get_bert_tokenizer()
        elif text_type.lower() == "roberta":
            self.tokenizer = get_roberta_tokenizer()
        elif text_type.lower() == "text":
            self.tokenizer = get_tokenizer("basic_english")
        else:
            self.tokenizer = get_distilbert_tokenizer()
        self.use_bert = True

        # get longest utt
        self.longest_utt = get_longest_utt(
            self.all_data, self.tokenizer, self.use_bert, text_type.lower() == "text"
        )

        # set DataPrep instance for each partition
        self.train_prep = DataPrep(
            self.d_type,
            self.all_data,
            self.longest_utt,
            text_type=text_type,
        )

        del self.all_data

        self.data = self.train_prep.combine_xs_and_ys(
            self.train_prep.data_tensors, self.speaker2idx
        )

    def get_data_folds(self):
        train_data, dev_data, test_data = create_data_folds_list(
            self.data, self.train_prop, self.test_prop
        )

        return train_data, dev_data, test_data


class DataPrep:
    """
    A class to prepare datasets
    Should allow input from meld, firstimpr, mustard, ravdess, cdc
    """

    def __init__(
        self,
        data_type,
        data,
        longest_utt,
        text_type="distilbert",
    ):
        # set data type
        self.d_type = data_type

        # get all used ids
        self.used_ids = data["message_id"].tolist()

        # use acoustic sets to get data tensors
        self.data_tensors = self.make_data_tensors(data, longest_utt, text_type)

        # get acoustic means
        self.acoustic_means = 0
        self.acoustic_stdev = 0

        # add pred type if needed
        self.pred_type = None

    def combine_xs_and_ys(self, data_tensors, speaker2idx):
        # combine all x and y data into list of tuples
        # if no gold labels, only combine x data
        data = []
        for i, item in enumerate(data_tensors["all_utts"]):
            data.append(
                {
                    "x_utt": item.clone().detach()
                    if type(item) == torch.tensor
                    else item,
                    "x_speaker": speaker2idx[data_tensors["all_speakers"][i]],
                    "ys": [
                        data_tensors["all_sentiments"][i],
                        data_tensors["all_emotions"][i],
                    ],
                    "audio_id": data_tensors["all_audio_ids"][i],
                    "utt_length": data_tensors["utt_lengths"][i],
                }
            )

        return data

    def make_data_tensors(self, text_data, longest_utt, text_type="distilbert"):
        """
        Make the data tensors for asist
        :param text_data: a pandas df containing text and gold
        :param longest_utt: length of longest utt
        :param text_type: the type of text or embeddings to get, if using
        :param use_text: whether to save raw text instead of embeddings
        :return: a dict containing tensors for utts, speakers, ys,
            and utterance lengths
        """
        # create holders for the data
        all_data = {
            "all_utts": [],
            "all_sentiments": [],
            "all_emotions": [],
            "all_speakers": [],
            "all_audio_ids": [],
            "utt_lengths": [],
        }

        # set class numbers for sent, emo
        sent2idx = {"positive": 2, "neutral": 1, "negative": 0}
        emo2idx = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }

        if text_type.lower() == "bert":
            emb_maker = BertEmb()
        elif text_type.lower() == "roberta":
            emb_maker = BertEmb(use_roberta=True)
        elif text_type.lower() == "distilbert":
            emb_maker = DistilBertEmb()

        for idx, row in tqdm(
            text_data.iterrows(), total=len(text_data), desc="Organizing data"
        ):
            # get audio id
            all_data["all_audio_ids"].append(row["message_id"])

            if text_type.lower() == "text":
                # get values from row
                # utt = tokenizer(clean_up_word(str(row['utt'])))
                utt = str(row["utt"])
                all_data["utt_lengths"].append(1)
                all_data["all_utts"].append(utt)
            else:
                # else use the bert/distilbert tokenizer instead
                utt, ids = emb_maker.tokenize(clean_up_word(str(row["utt"])))
                # convert ids to tensor
                ids = torch.tensor(ids)
                all_data["utt_lengths"].append(len(ids))
                # bert requires an extra dimension to match utt
                if text_type.lower() == "bert":
                    ids = ids.unsqueeze(0)
                utt_embs = emb_maker.get_embeddings(utt, ids, longest_utt)

                all_data["all_utts"].append(utt_embs)

            spk_id = row["participant"]
            sentiment = sent2idx[row["sentiment"]]
            emotion = emo2idx[row["emotion"]]

            all_data["all_speakers"].append(spk_id)
            all_data["all_sentiments"].append(sentiment)
            all_data["all_emotions"].append(emotion)

        # create pytorch tensors for each
        all_data["all_sentiments"] = torch.tensor(all_data["all_sentiments"])
        all_data["all_emotions"] = torch.tensor(all_data["all_emotions"])

        # pad and transpose utterance sequences
        if text_type != "text":
            all_data["all_utts"] = nn.utils.rnn.pad_sequence(all_data["all_utts"])
            all_data["all_utts"] = all_data["all_utts"].transpose(0, 1)

        # return data
        return all_data


def get_speaker_to_index_dict(speaker_set):
    """
    Take a set of speakers and return a speaker2idx dict
    speaker_set : the set of speakers
    """
    # set counter
    speaker_num = 0

    # set speaker2idx dict
    speaker2idx = {}

    # put all speakers in
    for speaker in speaker_set:
        speaker2idx[speaker] = speaker_num
        speaker_num += 1

    return speaker2idx


def create_data_folds_list(data, perc_train, perc_test, shuffle=True):
    """
    Create train, dev, and test data folds
    Specify the percentage that goes into each
    data: A LIST of the data that goes into all folds
    perc_* : the percentage for each fold
    Percentage not included in train or test fold allocated to dev
    shuffle: whether to shuffle the data
    """
    if shuffle:
        # shuffle the data
        random.shuffle(data)

    # get length
    length = len(data)

    # calculate proportion alotted to train and test
    train_len = math.floor(perc_train * length)
    test_len = math.floor(perc_test * length)

    # get datasets
    train_data = data[:train_len]
    test_data = data[train_len : train_len + test_len]
    dev_data = data[train_len + test_len :]

    # return data
    return train_data, dev_data, test_data
