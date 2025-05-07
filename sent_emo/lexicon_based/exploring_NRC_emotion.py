# exploring the relationship between the NRC emotion lexicon
# and the MultiCAT corpus

import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report

# check if punkt has been downloaded
# if not, run the following lines
import nltk

nltk.download("punkt")
nltk.download("wordnet")

from utils.baseline_utils import combine_modality_data


def load_train_data(load_path):
    # set paths to text data and gold labels
    text_base = f"{load_path}/MultiCAT/text_data"
    ys_base = f"{load_path}/MultiCAT/ys_data"

    # combine audio, text, and gold label pickle files into a Dataset
    train_text = pickle.load(open(f"{text_base}/train.pickle", "rb"))
    ys_train = pickle.load(open(f"{ys_base}/train.pickle", "rb"))

    train_data = combine_modality_data([train_text, ys_train])

    return train_data


def load_nrc_emotion(nrc_load_path):
    # load the file
    nrc_emo = pd.read_csv(
        f"{nrc_load_path}/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        sep="\t",
        header=None,
        names=["word", "emotion", "association"],
    )

    # association is either 0 or 1
    # we only care about words that have an emotion association
    nrc_emo = nrc_emo[nrc_emo.association != 0]

    # there are a total of 8 emotions plus two sentiments; they are:
    # anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    # positive, negative
    # we care about most of these, and the other two MAY be interesting
    # so let's include them all for now

    return nrc_emo


def load_nrc_emotion_as_dict(nrc_load_path):
    # load the file
    nrc_emo = pd.read_csv(
        f"{nrc_load_path}/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        sep="\t",
        header=None,
        names=["word", "emotion", "association"],
    )

    # association is either 0 or 1
    # we only care about words that have an emotion association
    nrc_emo = nrc_emo[nrc_emo.association != 0]

    # there are a total of 8 emotions plus two sentiments; they are:
    # anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    # positive, negative
    # we care about most of these, and the other two MAY be interesting
    # so let's include them all for now
    emo2idx = {"anger": 0,
               "anticipation": 1,
               "disgust": 2,
               "fear": 3,
               "joy": 4,
               "sadness": 5,
               "surprise": 6,
               "trust": 7,
               "positive": 8,
               "negative": 9
               }

    # get dict for emotions
    emo_wd_dict = {}
    for i, row in nrc_emo.iterrows():
        if row.word not in emo_wd_dict.keys():
            emo_wd_dict[row.word] = [emo2idx[row.emotion]]
        else:
            emo_wd_dict[row.word].append(emo2idx[row.emotion])

    return emo_wd_dict


def load_nrc_emotion_intensity(nrc_load_path):
    """
    Load the nrc emotion intensity lexicon
    """
    # load the file
    nrc_emo = pd.read_csv(
        f"{nrc_load_path}/NRC-Emotion-Intensity-Lexicon-v1.txt",
        sep="\t",
        header=None,
        names=["word", "emotion", "intensity"],
    )

    return nrc_emo


def calc_emotional_intensity_in_data(train_data, intensity_df):
    """
    Calculate the emotional intensity for the data
    """
    # set lemmatizer
    lemmatizer = WordNetLemmatizer()

    # get dict for emotions
    intensity_wd_dict = {}
    for i, row in intensity_df.iterrows():
        if row.word not in intensity_wd_dict.keys():
            intensity_wd_dict[row.word] = [(row.emotion, row.intensity)]
        else:
            intensity_wd_dict[row.word].append((row.emotion, row.intensity))

    emo_dict = {}

    # get ys to sentiment and emotion
    y2sent = {0: "negative", 1: "neutral", 2: "positive"}
    y2emo = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }

    for item in train_data:
        # tokenize the item
        tokenized = word_tokenize(item["x_utt"])

        lemmas = []

        # lemmatize the tokens
        for word in tokenized:
            lemma = lemmatizer.lemmatize(word)

            lemmas.append(lemma)

        # get emotion intensities for the utt
        emo_ints = calc_emotional_intensity_of_utt(lemmas, intensity_wd_dict)

        # add gold labels to emo_ints
        emo_ints["gold_sent"] = y2sent[int(item["ys"][0])]
        emo_ints["gold_emo"] = y2emo[int(item["ys"][1])]

        # add emo_ints as an item to emo_dict
        emo_dict[item["audio_id"]] = emo_ints

    return emo_dict


def calc_emotional_intensity_of_utt(utt, wd_intensity_dict):
    """
    Use the intensity of each word in the utt to calculate the
    avg intensity per emotion of the overall utt
    """
    # denominator for calculations
    denom = len(utt)

    # for each emotion, calculate the sum of intensities
    sums_dict = {
        "anger": 0.0,
        "anticipation": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "trust": 0.0,
        "positive": 0.0,
        "negative": 0.0,
    }  # adding positive and negative

    # split emotions into positive and negative
    # leave surprise out as it can be either
    positive_emos = ["anticipation", "joy", "trust"]
    negative_emos = ["anger", "digust", "fear", "sadness"]

    # go through each word in the utterance
    for wd in utt:
        # if it is in the intensity dict
        if wd in wd_intensity_dict.keys():
            # for each emotion - intensity tuple:
            for emo_int in wd_intensity_dict[wd]:
                # if item 0 == an emotion
                if emo_int[0] in sums_dict.keys():
                    # add to this emotion
                    sums_dict[emo_int[0]] += emo_int[1]
                    # check if it's positive or negative and add to that, too
                    if emo_int[0] in positive_emos:
                        sums_dict["positive"] += emo_int[1]
                    elif emo_int[0] in negative_emos:
                        sums_dict["negative"] += emo_int[1]

    # divide each item by the length of utt
    # to get avg for the utt
    # note: if we decide to just use max-class
    #   this is not necessary
    for k, v in sums_dict.items():
        sums_dict[k] = v / denom

    return sums_dict


def get_emo_vec(utterance, lemmatizer, emo_wd_dict):
    """
    Get a vector of emotion word counts from an utterance
    Vector is ordered as follows:
    [anger, anticipation, disgust, fear, joy, sadness,
    surprise, trust, positive, negative]
    """
    # tokenize the item
    tokenized = word_tokenize(utterance)

    lemmas = []

    # lemmatize the tokens
    for word in tokenized:
        lemma = lemmatizer.lemmatize(word)

        lemmas.append(lemma)

    # instantiate the utterance vector
    utt_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # compare with emo lexicon dict
    for lemma in lemmas:
        if lemma in emo_wd_dict.keys():
            for emo_wd_idx in emo_wd_dict[lemma]:
                utt_vec[emo_wd_idx] += 1

    return utt_vec


# we want to look at relationships between these, specifically:
# how many items have MORE words corresponding to that emotion than others
def calc_emo_words_with_data(train_data, emo_dataset):
    """
    Compare the number of emotion words with the outcomes of training data
    """
    # set lemmatizer
    lemmatizer = WordNetLemmatizer()

    # get dict for emotions
    emo_wd_dict = {}
    for i, row in emo_dataset.iterrows():
        if row.word not in emo_wd_dict.keys():
            emo_wd_dict[row.word] = [row.emotion]
        else:
            emo_wd_dict[row.word].append(row.emotion)

    emo_dict = {}

    # get ys to sentiment and emotion
    y2sent = {0: "negative", 1: "neutral", 2: "positive"}
    y2emo = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }

    for item in train_data:
        # add item id to emo dict
        # anger, anticipation, disgust, fear, joy, sadness, surprise, trust
        # positive, negative
        emo_dict[item["audio_id"]] = {
            "anger": 0,
            "anticipation": 0,
            "disgust": 0,
            "fear": 0,
            "joy": 0,
            "sadness": 0,
            "surprise": 0,
            "trust": 0,
            "positive": 0,
            "negative": 0,
            "gold_sent": y2sent[int(item["ys"][0])],
            "gold_emo": y2emo[int(item["ys"][1])],
        }

        # tokenize the item
        tokenized = word_tokenize(item["x_utt"])

        lemmas = []

        # lemmatize the tokens
        for word in tokenized:
            lemma = lemmatizer.lemmatize(word)

            lemmas.append(lemma)

        # compare with emo dataset
        for lemma in lemmas:
            if lemma in emo_wd_dict.keys():
                for emo_wd in emo_wd_dict[lemma]:
                    emo_dict[item["audio_id"]][emo_wd] += 1

    return emo_dict


def convert_emo_dict_to_df(emotion_dict):
    reordered_dict = {
        "audio_id": [],
        "anger": [],
        "anticipation": [],
        "disgust": [],
        "fear": [],
        "joy": [],
        "sadness": [],
        "surprise": [],
        "trust": [],
        "positive": [],
        "negative": [],
        "gold_sent": [],
        "gold_emo": [],
    }

    for item in emotion_dict.keys():
        reordered_dict["audio_id"].append(item)
        reordered_dict["anger"].append(emotion_dict[item]["anger"])
        reordered_dict["anticipation"].append(emotion_dict[item]["anticipation"])
        reordered_dict["disgust"].append(emotion_dict[item]["disgust"])
        reordered_dict["fear"].append(emotion_dict[item]["fear"])
        reordered_dict["joy"].append(emotion_dict[item]["joy"])
        reordered_dict["sadness"].append(emotion_dict[item]["sadness"])
        reordered_dict["surprise"].append(emotion_dict[item]["surprise"])
        reordered_dict["trust"].append(emotion_dict[item]["trust"])
        reordered_dict["positive"].append(emotion_dict[item]["positive"])
        reordered_dict["negative"].append(emotion_dict[item]["negative"])
        reordered_dict["gold_sent"].append(emotion_dict[item]["gold_sent"])
        reordered_dict["gold_emo"].append(emotion_dict[item]["gold_emo"])

    items_df = pd.DataFrame.from_dict(reordered_dict)

    return items_df


def compare_emo_items_in_data(data_df):
    # get ys to sentiment and emotion
    sent2y = {"negative": 0, "neutral": 1, "positive": 2}
    emo2y = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6,
    }

    # perform initial comparisons on number of items with x as best
    # look for the max values in each row
    # if they are whole numbers, just look for maxes
    if data_df["positive"].dtypes == "int64":
        mxs = data_df.eq(data_df.max(axis=1), axis=0)
        # join the column names of the max values of each row into a single string
        data_df["Max"] = mxs.dot(mxs.columns + ", ").str.rstrip(", ")
    # else look for max of sent AND max of emo
    else:
        mxs0 = get_max_sent(data_df)
        mxs1 = get_max_emo(data_df)

        data_df["Max"] = mxs0 + ", " + mxs1

    # get predicted sentiments and emotions
    data_df["pred_sent"] = data_df["Max"].apply(lambda x: get_pred_sent(x))
    data_df["pred_sent"] = data_df["pred_sent"].apply(lambda x: sent2y[x])

    data_df["pred_emo"] = data_df["Max"].apply(lambda x: get_pred_emo(x))
    data_df["pred_emo"] = data_df["pred_emo"].apply(lambda x: emo2y[x])

    data_df["gold_sent"] = data_df["gold_sent"].apply(lambda x: sent2y[x])
    data_df["gold_emo"] = data_df["gold_emo"].apply(lambda x: emo2y[x])

    # # convert to lists of numbers for confusion matrix
    # all_pred_sents = [sent2y[item] for item in data_df['pred_sent'].tolist()]
    # all_pred_emos = [emo2y[item] for item in data_df['pred_emo'].tolist()]
    #
    # # convert gold labels
    # all_gold_sents = [sent2y[item] for item in data_df['gold_sent'].tolist()]
    # all_gold_emos = [emo2y[item] for item in data_df['gold_emo'].tolist()]
    #
    # # get confusion matrix of emos
    # print(confusion_matrix(all_gold_emos, all_pred_emos))
    # print(classification_report(all_gold_emos, all_pred_emos))
    #
    # # get confusion matrix of sents
    # print(confusion_matrix(all_gold_sents, all_pred_sents))
    # print(classification_report(all_gold_sents, all_pred_sents))

    return data_df


def get_max_sent(df):
    df2 = df[["positive", "negative"]]
    return df2.apply(
        lambda x: get_max_or_neutral(x, threshold=0.0), axis=1
    )  # 0.05 when alone


def get_max_emo(df):
    df2 = df[
        [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "trust",
        ]
    ]
    return df2.apply(
        lambda x: get_max_or_neutral(x, threshold=0.0), axis=1
    )  # 0.01 when alone


def get_max_or_neutral(row, threshold=0.0):
    """
    Get the max of the options if any
    If all items are 0 (or below a threshold), return 'neutral'
    """
    if "positive" in row.index:
        if row.loc["positive"] <= threshold and row.loc["negative"] <= threshold:
            return "neutral"
        else:
            return row.idxmax()
    else:
        if (
            (row.loc["anger"] <= threshold)
            and (row.loc["anticipation"] <= threshold)
            and (row.loc["disgust"] <= threshold)
            and (row.loc["fear"] <= threshold)
            and (row.loc["joy"] <= threshold)
            and (row.loc["sadness"] <= threshold)
            and (row.loc["surprise"] <= threshold)
            and (row.loc["trust"] <= threshold)
        ):
            return "neutral"
        else:
            return row.idxmax()


def get_pred_sent(maxes_string):
    """Get predicted sentiment from a list of maxes as a string"""
    maxes = maxes_string.split(", ")
    # assumes pos and neg cancel each other out
    # this is just an initial prediction
    # if we bias towards pos or neg, results are much worse
    if "positive" in maxes and "negative" in maxes:
        return "neutral"
    elif "positive" in maxes:
        return "positive"
    elif "negative" in maxes:
        return "negative"
    else:
        return "neutral"


def get_pred_emo(maxes_string):
    """Get predicted emotion from a list of maxes as a string"""
    maxes = maxes_string.split(", ")
    emos = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    if len(maxes) == 1:
        if maxes[0] in emos:
            return maxes[0]
        else:
            return "neutral"
    else:
        num_emos = 0
        this_emo = ""
        for max_emo in maxes:
            if max_emo in emos:
                num_emos += 1
                if num_emos == 1:
                    this_emo = max_emo
                else:
                    this_emo = "neutral"
        if this_emo != "":
            return this_emo
        else:
            return "neutral"


def compare_emo_and_intensity_items_in_data(df1, df2):
    """
    Use both raw words and intensity words to calculate sent/emo for an utterance
    Make predictions based on these
    """
    # combine df1 and df2
    df = df1.merge(df2, how="inner", on=["audio_id"])

    # use these to vote
    df["pred_sent"] = df.apply(lambda x: vote_for_sentiment(x), axis=1)
    df["pred_emo"] = df.apply(lambda x: vote_for_emotion(x), axis=1)

    # get confusion matrics and evaluation metrics
    # # get confusion matrix of emos
    print(confusion_matrix(df["gold_emo_x"].tolist(), df["pred_emo"].tolist()))
    print(classification_report(df["gold_emo_x"].tolist(), df["pred_emo"].tolist()))
    #
    # # get confusion matrix of sents
    print(confusion_matrix(df["gold_sent_x"].tolist(), df["pred_sent"].tolist()))
    print(classification_report(df["gold_sent_x"].tolist(), df["pred_sent"].tolist()))


def vote_for_sentiment(row):
    """
    Given a row, vote for sentiment
    The intensity scores are expected to be in {name}_x cols
    The counts are expected to be in {name}_y cols
    """
    # if they vote for the same thing, choose this
    if row["pred_sent_x"] == row["pred_sent_y"]:
        return row["pred_sent_x"]
    # if using intensities is neutral, return neutral
    elif row["pred_sent_x"] == 1:
        return 1
    # if using word counts is neutral
    elif row["pred_sent_y"] == 1:
        # if intensity is non-trivially negative, return negative
        if row["pred_sent_x"] == 0 and row["negative_x"] > 0.025:
            return 0
        # if intensity is non-trivially positive, return positive
        elif row["pred_sent_x"] == 2 and row["positive_x"] > 0.025:
            return 2
        # if intensity is low, return neutral
        else:
            return 1
    # if conflicting guesses are present, return neutral
    else:
        return 1

    # # if they vote for the same thing, choose this
    # if item1 == item2:
    #     return item1
    # # if only one is non-neutral, choose this
    # elif item1 == 1:
    #     return item2
    # elif item2 == 1:
    #     return item1
    # # else, return neutral
    # else:
    #     return 1


def vote_for_emotion(row):
    return 4


# def vote_for_emotion(item1, item2):
#     """
#     Given 2 items, vote for emotion
#     """
#     # if the same, return it
#     if item1 == item2:
#         return item1
#     # if one is neutral, return the other
#     elif item1 == 4:
#         return item2
#     elif item2 == 4:
#         return item1
#     # else, return neutral
#     # todo: try out a more complicated logic here
#     else:
#         # if item1 == 3 or item2 == 3:
#         #     return 3
#         # elif item1 == 5 or item2 == 5:
#         #     return 5
#         # else:
#         return 4


if __name__ == "__main__":
    load_path = "data"
    emo_load_path = "~/datasets/NRC-Emotion-Intensity-Lexicon"
    emo_load_path2 = "~/datasets/NRC-Emotion-Lexicon"

    train_data = load_train_data(load_path)

    emo_data = load_nrc_emotion_intensity(emo_load_path)
    data_dict = calc_emotional_intensity_in_data(train_data, emo_data)

    data_df = convert_emo_dict_to_df(data_dict)

    data_df = compare_emo_items_in_data(data_df)

    emo_data2 = load_nrc_emotion(emo_load_path2)
    data_dict2 = calc_emo_words_with_data(train_data, emo_data)
    data_df2 = convert_emo_dict_to_df(data_dict2)

    data_df2 = compare_emo_items_in_data(data_df2)

    compare_emo_and_intensity_items_in_data(data_df, data_df2)