import sqlite3
import os
import pandas as pd
import shutil
import regex as re

train_files = [
    "T000603",
    "T000604",
    "T000611",
    "T000612",
    "T000620",
    "T000622",
    "T000623",
    "T000624",
    "T000627",
    "T000628",
    "T000631",
    "T000632",
    "T000635",
    "T000636",
    "T000637",
    "T000638",
    "T000703",
    "T000704",
    "T000713",
    "T000714",
    "T000715",
    "T000716",
    "T000719",
    "T000720",
    "T000723",
    "T000724",
    "T000729",
    "T000730",
]
val_files = ["T000613", "T000607", "T000608", "T000633", "T000634"]
test_files = [
    "T000605",
    "T000606",
    "T000671",
    "T000672",
    "T000625",
    "T000626",
    "T000727",
    "T000728",
    "T000737",
    "T000738",
    "T000609",
    "T000610",
]

all_files = train_files + val_files + test_files

splits = {"train": train_files, "val": val_files, "test": test_files}


def get_basic_da(tag: str):
    spl = tag.split("|")
    for value in spl:  # get rid of rt, d. When floor mechanism at start of pipe DA, followed by another DA, skip floor mechanism
        value = re.sub("\.(%--|%-|%|x)", "", value)
        value = re.sub("\^rt", "", value)  # getting rid of rt tags
        # merge labels
        value = re.sub("^(%--|%-|%)", "%", value)  # x, %, %-, %--
        value = re.sub("\^bk(?![a-z])", "^aa", value)  # getting rid of d tags

        if "^" in value:
            value = value.split("^")[0]  # take general tag

        value = value.strip()
        if value == "s":
            return "statement"
        elif value in ["fg", "fh", "h"]:
            return "fillers"
        elif value in ["%"]:
            return "disruptions"
        elif value in ["b"]:
            return "backchannels"
        elif value in ["qy", "qw", "qr", "qrr", "qo", "qh"]:
            return "questions"
        elif value in ["x", "z"]:
            return "unlabelable"
        assert False


def load_data(filepath: str, new_dir: str):
    # filepath: path to the database
    # new_dir: path to new processed data

    df = sqlite3.connect(filepath)
    df = pd.read_sql_query("SELECT * FROM utterance", df)
    mask = df["trial"].isin(all_files)
    data = df[mask]

    for split in splits:
        dirname = os.path.join(new_dir, split)
        if os.path.exists(dirname) and os.path.isdir(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

        conv_numb = 0
        for conv in splits[split]:
            conv_data = data[data["trial"] == conv]
            fileName = conv + ".txt"
            fileOpen = open(os.path.join(dirname, fileName), "w")

            for index, row in conv_data.iterrows():
                utt = row["asr_text"]
                corr_utt = row["corrected_text"]
                participant = row["participant"]
                label = row["dialog_acts"]  # label

                if pd.isnull(utt):
                    if (label == "x" or label == "z") and pd.isnull(corr_utt):
                        continue
                    if pd.isnull(corr_utt):
                        continue
                    assert not pd.isnull(corr_utt)
                if not pd.isnull(corr_utt):
                    utt = corr_utt

                utt_split = utt.split("|")
                tag = label.split(":")
                tag = tag[0]  # for Quotes: DA on left is for entire utt
                spl = tag.split("|")
                assert len(spl) == len(utt_split)
                for i in range(len(utt_split)):
                    spl_utt = utt_split[i]
                    tag_basic = get_basic_da(spl[i])
                    tag_basic = tag_basic.strip()

                    if tag_basic not in ["z", "x", "unlabelable"]:
                        fileOpen.write(
                            "{}|{}|{}\n".format(
                                participant,
                                spl_utt.replace("|", ""),
                                tag_basic,
                            )
                        )
            conv_numb += 1

            fileOpen.close()


load_data("../data_files/multicat2.db", "ARROct24/DA")
