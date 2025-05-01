import pandas as pd
import sys, os
from pathlib import Path
import re


class ToMCATSentEmoPrep:
    def __init__(self, base_dir):
        # the directory within which to work
        self.base_dir = base_dir

        # holder for tipi results
        self.tipi_dir = f"{base_dir}/tipi"
        # holder for sent-emo annotated files
        self.annotations_dir = f"{base_dir}/sent-emo"

        # holder for corrected transcriptions
        self.correct_times_dir = f"{base_dir}/corrected_transcriptions"

    def combine_corrected_and_annotated(self):
        # 2023-06-22: This function may not be needed.
        # go through items that exist for corrected transcriptions
        for item in os.listdir(self.annotations_dir):
            if item.endswith(".csv"):
                # get name of csv
                csv_name = item.rsplit("/", 1)[0]

                # open as pandas df
                item_df = pd.read_csv(f"{self.annotations_dir}/{item}")
                item_df = item_df[
                    ["participant", "utt", "emotion", "sentiment", "start_timestamp"]
                ]

                # check for equivalent item in corrected transcriptions
                if os.path.isfile(f"{self.correct_times_dir}/{csv_name}"):
                    print(f"{self.correct_times_dir}/{csv_name}")
                    sent_df = pd.read_csv(f"{self.correct_times_dir}/{csv_name}")
                    sent_df = sent_df[
                        [
                            "participant",
                            "utt",
                            "corr_utt",
                            "start_timestamp",
                            "end_timestamp",
                        ]
                    ]
                    sent_df = sent_df[sent_df["utt"].notna()]

                    # combine items when there is an annotation for the utt
                    # join should probably be inner
                    combined_df = pd.merge(
                        item_df, sent_df, on=["participant", "utt"], how="left"
                    )
                    # combined_df = item_df.join(sent_df, on=["participant", "utt"])

                    # either save combined df or keep it in memory to deal with later
                    combined_df.to_csv(
                        f"{self.base_dir}/combined/{csv_name}", index=False
                    )

    def combine_files(self):
        """
        Combine files of corrected/annotated data
        """
        all_files = None
        # go through items that exist for corrected transcriptions
        for item in os.listdir(self.annotations_dir):
            if item.endswith(".csv"):
                print(item)
                # get name of csv
                csv_name = item.rsplit("/", 1)[0]

                csv_name_info = csv_name.split("_")
                trial_id = csv_name_info[3][-7:]
                team_id = csv_name_info[4][-8:]

                data = pd.read_csv(f"{self.annotations_dir}/{item}")

                data["trial_id"] = trial_id
                data["team_id"] = team_id

                data = self._remove_unlabeled(data)
                data = self._fix_labels(data)

                if all_files is None:
                    all_files = data
                else:
                    all_files = pd.concat([all_files, data], axis=0)

        # todo: change to if saving required
        all_files.to_csv(f"{self.base_dir}/all_sent-emo.csv", index=False)

        return all_files

    def _remove_unlabeled(self, df):
        """
        Remove the rows without labels for sentiment/emotion
        :param df:
        :return:
        """
        df.dropna(subset=["emotion", "sentiment"], inplace=True)

        df = df[(df["emotion"] != "0") & (df["sentiment"] != "0")]

        return df

    def _fix_labels(self, df):
        df["emotion"] = df["emotion"].apply(lambda x: str(x).strip())
        df["sentiment"] = df["sentiment"].apply(lambda x: str(x).strip())

        errors = {
            "angry": "anger",
            "neural": "neutral",
            "neutrl": "neutral",
            "suprise": "surprise",
            "sadeness": "sadness",
            "netural": "neutral",
            "negatve": "negative",
            "positie": "positive",
            "negatie": "negative",
            "poitive": "positive",
            "nutral": "neutral",
            "neative": "negative",
            "postive": "positive",
            "postivie": "positive",
            "n": "neutral",
            "netral": "neutral",
            "negtive": "negative",
        }

        df["emotion"] = df["emotion"].apply(
            lambda x: errors[x] if x in errors.keys() else x
        )
        df["sentiment"] = df["sentiment"].apply(
            lambda x: errors[x] if x in errors.keys() else x
        )

        return df

    def read_in_tipi(self):
        all_tipi = None
        for item in os.listdir(self.tipi):
            tipi_df = pd.read_csv(f"{self.tipi}/{item}")
            # check if there is already a tipi df
            if all_tipi == None:
                all_tipi = tipi_df
            else:
                # join them together if needed
                all_tipi = pd.join(tipi_df, all_tipi)

    def count_all_emotions(self, all_data):
        """
        Get class counts for all emotion classes in all the data
        all_data : a pandas df containing the data
        """
        return all_data["emotion"].value_counts()

    def count_all_sentiments(self, all_data):
        """
        Get class counts for all sentiment classes in all the data
        """
        return all_data["sentiment"].value_counts()

    def count_all_traits(self, all_data):
        """
        Get class counts for all personality trait classes in the data
        """
        return all_data["trait"].value_counts()


class ToMCATDatasetPrep:
    def __init__(self, basedir):
        self.base = Path(basedir)
        self.sentemo_dir = self.base / "sent-emo"
        self.da_dir = self.base / "updated_da"
        self.merged = self._merge_files()

    def _merge_files(self):
        """
        Merge all possible files
        :return: dict of fname: df for combined da and sent-emo files
        """
        merged = {}

        da_names = [f.name for f in self.da_dir.iterdir() if f.is_file()]

        for f in self.sentemo_dir.iterdir():
            name = "_".join(f.name.split("_")[1:])
            # We have some files labeled Vers-1 and some labeled with Vers-6.
            # We do this step below since there is at least one instance where
            # sent/emo and da files have different versions.
            # TODO: Test if this breaks things.
            if "T0006" in name and "Vers-1" in name:
                re.sub("Vers-1", "Vers-6", name)
            # print(name)

            # get the sent-emo file
            sentemo_df = pd.read_csv(f , usecols=["utt", "participant", "message_id", "sentiment", "emotion"])

            # see if there is a corresponding file
            # with da annotations
            if name in da_names:
                # read in da annotations file
                da = pd.read_csv(self.da_dir / name)

                # cannot use df.merge because of multiple identical rows without message_id
                da["sentiment"] = None
                da["emotion"] = None

                # ordered_utt = sentemo_df["utt"].tolist()
                ordered_sent = sentemo_df["sentiment"].tolist()
                ordered_emo = sentemo_df["emotion"].tolist()
                ordered_msg = sentemo_df["message_id"].tolist()

                for row in da.itertuples():
                    if row.message_id in ordered_msg:
                        idx = ordered_msg.index(row.message_id)
                        da.at[row.Index, "sentiment"] = ordered_sent[idx]
                        da.at[row.Index, "emotion"] = ordered_emo[idx]
                merged[name] = da
            else:
                # if no da annotations
                # just save sentemo
                # todo: verify that this works
                merged[name] = sentemo_df

        return merged

    def convert_values(self, conversion_dict):
        """
        Convert playernames to participant IDs
        :param conversion_dict:
        :return:
        """
        for k, v in self.merged.items():
            trial = k.split("_")[2].split("-")[-1]
            if trial not in conversion_dict.keys():
                # Not sure why we have these special cases...
                if trial == "T000607":
                    conv = conversion_dict["T000608"]
                elif trial == "T000633":
                    conv = conversion_dict["T000634"]
                else:
                    exit(f"Trial {trial} not found")
            else:
                conv = conversion_dict[trial]
            v["participant"] = v["participant"].apply(lambda x: conv[x])

    def save_merged(self):
        for k, v in self.merged.items():
            v.to_csv(self.base / f"combined/{k}", index=False)


if __name__ == "__main__":
    # # read in files
    # base = "/media/jculnan/backup/jculnan/datasets/asist_data2"
    #
    # prep_obj = ToMCATSentEmoPrep(base)
    #
    # #prep_obj.combine_corrected_and_annotated()
    # all_files = prep_obj.combine_files()
    # emos = prep_obj.count_all_emotions(all_files)
    # sents = prep_obj.count_all_sentiments(all_files)
    # print("Emotion counts by class: ")
    # print(emos)
    # print("Sentiment counts by class: ")
    # print(sents)
    pass
