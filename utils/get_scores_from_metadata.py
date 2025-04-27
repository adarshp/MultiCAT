from file_read_backwards import FileReadBackwards
import json
from pathlib import Path
import pandas as pd


def get_participant_info(json_path):
    """
    Get the participant info out of a single json file
    :param json_file:
    :return:
    """
    print(json_path)
    with FileReadBackwards(json_path, encoding="utf-8") as json_file:
        # get trial number
        trial = str(json_path).split("/")[-1].split("_")[2].split("-")[-1]
        participant_info = [trial, ""]
        # find the last mention of team_score_agg
        for l in json_file:
            # if line is the metadata line with IDs
            theline = json.loads(l)

            if "team_score_agg" in theline["data"].keys():
                score = theline["data"]["team_score_agg"]

                participant_info = [trial, score]

                break

    return participant_info


def get_series_of_participant_info(metadata_dir):
    """
    Using a path to a directory containing metadata files
    Go through each file and get the final score for that trial
    :param metadata_dir:
    :return:
    """
    meta_path = Path(metadata_dir)

    all_scores = []
    for metadata in meta_path.iterdir():
        if metadata.suffix == ".metadata":
            score = get_participant_info(metadata)

            all_scores.append(score)

    return all_scores


def save_scores(list_of_scores, save_path=None):
    scores = pd.DataFrame(list_of_scores, columns=["Trial_ID", "Score"])
    if save_path:
        scores.to_csv(save_path, index=False)
    else:
        scores.to_csv(
            "/media/jculnan/backup/jculnan/datasets/asist_data2/scores.csv", index=False
        )


if __name__ == "__main__":
    # meta_path = "/media/jculnan/backup/jculnan/datasets/asist_data2/metadata"
    meta_path = "/media/jculnan/datadrive/asist_data_copy/metadata"

    scores = get_series_of_participant_info(meta_path)
    save_scores(scores, save_path="/media/jculnan/datadrive/asist_data_copy/scores.csv")
