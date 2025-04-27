# get participant information from the .metadata files
import json
import logging
from pathlib import Path
from tqdm import tqdm
from glob import glob
from itertools import chain
from tqdm.contrib.concurrent import process_map


class MetadataToParticipantInfo:
    def __init__(self, base_path):
        # path to metadata files
        self.base_path = base_path

    def _get_participant_info(self, filepath):
        """
        Get the participant info out of a single json file
        :param json_file:
        :return:
        """
        participant_info = []

        trial = filepath.rpartition("Trial-")[2].split("_")[0]
        team = filepath.rpartition("Team-")[2].split("_")[0]
        condbtwn = filepath.rpartition("CondBtwn-")[2].split("_")[0]
        condwin = filepath.rpartition("CondWin-")[2].split("_")[0]
        with open(filepath) as f:
            # first line is a header line but doesn't contain
            # all the required information
            f.readline()
            for line in f:
                # if line is the metadata line with IDs
                message = json.loads(line)
                # if "header" in theline.keys() and "name" in theline["data"].keys():
                if (
                    message["topic"] == "trial"
                    and message["msg"]["sub_type"] == "start"
                ):
                    client_info = message["data"]
                    players = message["data"]["client_info"]
                    for player in players:
                        p_id = player["participant_id"]
                        p_name = player["playername"]

                        participant_info.append([team, trial, p_id, p_name])

                    break

        return participant_info

    def get_info_on_multiple_trials(self):
        """
        Get information about multiple trials
        All trials' metadata should be stored in the same
            directory
        :param metadata_dir_path:
        :return:
        """
        all_participant_info = list(
            chain(
                *process_map(
                    self._get_participant_info,
                    glob(self.base_path + "/HSRData*Trial-T0*.metadata"),
                    max_workers=100,
                )
            )
        )
        # for f in tqdm(glob(self.base_path+"/HSRData*Trial-T0*.metadata")):
        # p_info = self._get_participant_info(str(f))
        # all_participant_info.extend(p_info)
        # all_participant_info = list(chain(*r))

        return all_participant_info

    def return_trial_player_dict(self, participant_info):
        """
        Return a dict of Trial_ID: {Playername: Participant ID}
        Used in conjunction with data compilation utils
        To switch out player names for participant IDs
        :return:
        """
        tp_dict = {}
        if type(participant_info[0] == list):
            for trial in participant_info:
                if trial[1] not in tp_dict.keys():
                    tp_dict[trial[1]] = {trial[3]: trial[2]}
                else:
                    tp_dict[trial[1]][trial[3]] = trial[2]

        return tp_dict

    def save_participant_info(self, participant_info, save_location):
        """
        Save extracted participant information
        :param save_location: path + name of file to save participant info
        :return:
        """
        participant_info.to_csv(save_location, index=False)


def add_scores_to_participant_info(participant_df, scores_df):
    """
    Add scores from mission to the participant data
    :param scores_df:
    :return:
    """
    participant_df["Score"] = participant_df["Trial_ID"].map(
        scores_df.set_index("Trial_ID")["Score"]
    )

    return participant_df


if __name__ == "__main__":
    import pandas as pd

    base_path = f"/media/jculnan/backup/jculnan/datasets/asist_data2"
    participant_info = f"{base_path}/participant_info.csv"
    scores = f"{base_path}/scores.csv"

    part = pd.read_csv(participant_info)
    sc = pd.read_csv(scores)

    print(add_scores_to_participant_info(part, sc))
