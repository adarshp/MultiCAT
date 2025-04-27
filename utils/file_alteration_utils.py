from pathlib import Path
import pandas as pd
import re


def update_vers3_file_sentemo(vers3_df, vers6_df, saved_name):
    """
    Adding corrected information to annotations done on a vers-3 file
    Note: we cannot simply merge the two dfs because there are
    multiple instances of the same utterances, and this throws
    off the results--we need to preserve the order of utterances
    :param vers3_df:
    :param vers6_df:
    :return:
    """
    vers3_short = vers3_df[["utt", "sentiment", "emotion"]]
    vers3_short = vers3_short[~vers3_short["sentiment"].isna()]

    ordered_utt = vers3_short["utt"].tolist()
    ordered_sent = vers3_short["sentiment"].tolist()
    ordered_emo = vers3_short["emotion"].tolist()

    vers6_df["sentiment"] = None
    vers6_df["emotion"] = None

    position = 0
    for row in vers6_df.itertuples():
        if row.utt == ordered_utt[position]:
            vers6_df.set_value(row.Index, "sentiment", ordered_sent[position])
            vers6_df.set_value(row.Index, "emotion", ordered_emo[position])

            if position < len(ordered_utt) - 1:
                position += 1
            else:
                break

    vers6_df.to_csv(saved_name, index=False)


def update_vers3_file_da(vers3_df, vers6_df, saved_name):
    """
    Adding corrected information to annotations done on a vers-3 file
    Note: we cannot simply merge the two dfs because there are
    multiple instances of the same utterances, and this throws
    off the results--we need to preserve the order of utterances
    :param vers3_df:
    :param vers6_df:
    :return:
    """
    vers3_short = vers3_df[["utt", "sentiment", "emotion"]]
    vers3_short = vers3_short[~vers3_short["sentiment"].isna()]

    ordered_utt = vers3_short["utt"].tolist()
    ordered_sent = vers3_short["sentiment"].tolist()
    ordered_emo = vers3_short["emotion"].tolist()

    vers6_df["sentiment"] = None
    vers6_df["emotion"] = None

    position = 0
    for row in vers6_df.itertuples():
        if row.utt == ordered_utt[position]:
            vers6_df.set_value(row.Index, "sentiment", ordered_sent[position])
            vers6_df.set_value(row.Index, "emotion", ordered_emo[position])

            if position < len(ordered_utt) - 1:
                position += 1
            else:
                break

    vers6_df.to_csv(saved_name, index=False)


def update_multiple_vers3_files(path_to_vers3, path_to_vers6, kind="da"):
    """
    Update a group of vers3 files
    :param path_to_vers3:
    :param path_to_vers6:
    :return:
    """
    vers3 = Path(path_to_vers3)
    for item in vers3.iterdir():
        if item.suffix == ".csv":
            name = item.name
            path = item.parents[0]
            vers3_df = pd.read_csv(item)
            v6_name = re.sub("Vers-3", "Vers-6", name)
            v3_annotator = v6_name.split("_")[0]
            v6_name = "_".join(v6_name.split("_")[1:])
            vers6_df = pd.read_csv(f"{path_to_vers6}/{v6_name}")

            if kind == "da":
                update_vers3_file_da(
                    vers3_df, vers6_df, f"{path}/{v3_annotator}_{v6_name}"
                )
            else:
                update_vers3_file_sentemo(
                    vers3_df, vers6_df, f"{path}/{v3_annotator}_{v6_name}"
                )


if __name__ == "__main__":
    base_path = "/media/jculnan/backup/jculnan/datasets/asist_data2"
    v3_path = f"{base_path}/v3"
    v6_path = f"/home/jculnan/asist_data/test-csv"

    update_multiple_vers3_files(v3_path, v6_path)
