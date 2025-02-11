import os
import pandas as pd
import regex as re
import shutil
import sqlite3


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
label_map_s = {
    "s": 0,
    "aa": 1,
    "%": 2,
    "df": 3,
    "b": 4,
    "t": 5,
    "ba": 6,
    "fg": 7,
    "g": 8,
    "na": 9,
    "cs": 10,
    "fh": 11,
    "co": 12,
    "bu": 13,
    "tc": 14,
    "e": 15,
    "qy": 16,
    "nd": 17,
    "fe": 18,
    "bs": 19,
    "no": 20,
    "qw": 21,
    "ng": 22,
    "bsc": 23,
    "qrr": 24,
    "aap": 25,
    "2": 26,
    "j": 27,
    "qr": 28,
    "ar": 29,
    "h": 30,
    "qh": 31,
    "m": 32,
    "f": 33,
    "r": 34,
    "fa": 35,
    "am": 36,
    "cc": 37,
    "qo": 38,
    "t1": 39,
    "bd": 40,
    "d": 41,
    "bh": 42,
    "arp": 43,
    "by": 44,
    "bc": 45,
    "t3": 46,
    "br": 47,
    "ft": 48,
    "fw": 49,
}
label_map_b = {
    "statement": 0,
    "fillers": 1,
    "disruptions": 2,
    "backchannels": 3,
    "questions": 4,
    "unlabelable": 5,
}


def load_data(filepath: str, new_dir: str, basic_tags: bool = False):
    label_map = label_map_b if basic_tags else label_map_s
    df = sqlite3.connect(filepath)
    df = pd.read_sql_query("SELECT * FROM utterance", df)
    mask = df["trial"].isin(all_files)
    data = df[mask]

    if os.path.exists(new_dir) and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    for split in splits:
        fileName = split + ".csv"
        fileOpen = open(os.path.join(new_dir, fileName), "w")
        fileOpen.write("speaker|switch|text|act|conv_id\n")

        conv_numb = 0
        for conv in splits[split]:
            conv_data = data[data["trial"] == conv]

            last_spk = None
            spk_id = 0
            for index, row in conv_data.iterrows():
                utt = row["asr_text"]
                corr_utt = row["corrected_text"]
                participant = row["participant"]
                label = row["dialog_acts"]

                if participant != last_spk:
                    last_spk = participant
                    if spk_id == 0:
                        spk_id = 1
                    else:
                        spk_id = 0

                if pd.isnull(utt):
                    if (label == "x" or label == "z") and pd.isnull(corr_utt):
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
                    tag_f = (
                        get_basic_da(spl[i]) if basic_tags else get_single_da(spl[i])
                    )
                    tag_f = tag_f.strip()

                    if tag_f not in ["z", "x", "unlabelable"]:
                        fileOpen.write(
                            "{}|{}|{}|{}|{}\n".format(
                                str(spk_id),
                                "",
                                spl_utt,
                                label_map[tag_f],
                                conv_numb,
                            )
                        )

            conv_numb += 1

        fileOpen.close()


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


def get_single_da(tag):
    spl = tag.split("|")
    for value in spl:  # get rid of rt, d. When floor mechanism at start of pipe DA, followed by another DA, skip floor mechanism
        value = re.sub("\.(%--|%-|%|x)", "", value)
        value = re.sub("\^rt", "", value)  # getting rid of rt tags
        # merge labels
        value = re.sub("^(%--|%-|%)", "%", value)  # x, %, %-, %--
        value = re.sub("\^bk(?![a-z])", "^aa", value)  # getting rid of d tags

        if "^" in value:
            value = value.split("^")[1]
        return value


load_data("../data_files/multicat2.db", "ARROct24/DA_csvs", False)
