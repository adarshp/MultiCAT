# used for splitting large audio files into utterance-length files
import os
import subprocess as sp
from pathlib import Path
import pandas as pd
import re


def extract_portions_of_mp4_or_wav(
    path_to_sound_file,
    sound_file,
    start_time,
    end_time,
    save_path=None,
    short_file_name=None,
):
    """
    Extracts only necessary portions of a sound file
    sound_file : the name of the full file to be adjusted
    start_time : the time at which the extracted segment should start
    end_time : the time at which the extracted segment should end
    short_file_name : the name of the saved short sound file
    """
    # set full path to file
    full_sound_path = os.path.join(path_to_sound_file, sound_file)

    # check sound file extension
    if not short_file_name:
        print("short file name not found")
        short_file_name = f"{sound_file.split('.')[0]}_{start_time}_{end_time}.wav"

    if save_path is not None:
        save_name = f"{save_path}/{short_file_name}"
    else:
        save_name = f"{path_to_sound_file}/{short_file_name}"

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    # get shortened version of file
    if not os.path.exists(save_name):
        sp.run(
            [
                "ffmpeg",
                "-i",
                full_sound_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                str(save_name),
            ]
        )

    return save_name


def split_wav_into_utterances(path_to_sound_file, sound_file, wav_level_df):
    """
    Take a series of wav files and split into a group of utterance-level wav files
    Must use a separate wav file for each speaker in the group
    :param path_to_sound_file: the path to the sound file
    :param sound_file: the wav file to be split
    :param wav_level_df: A dataframe including timestamps and utterance IDs
    :return:
    """
    wav_level_df.dropna(subset=["start_timestamp", "end_timestamp"], inplace=True)
    if len(wav_level_df) > 0:
        # add 250ms to start and end timestamps
        # to ensure we don't have any completely empty items
        # due to the timestamp rounding Google uses
        wav_level_df["start_timestamp"] = wav_level_df["start_timestamp"].apply(
            lambda x: convert_timestring_to_seconds(x)
        )
        wav_level_df["start_timestamp"] = wav_level_df["start_timestamp"] - 0.250
        wav_level_df["end_timestamp"] = wav_level_df["end_timestamp"].apply(
            lambda x: convert_timestring_to_seconds(x)
        )
        wav_level_df["end_timestamp"] = wav_level_df["end_timestamp"] + 0.250

        row_names = wav_level_df["message_id"].tolist()
        start_times = wav_level_df["start_timestamp"].tolist()
        end_timestamp = wav_level_df["end_timestamp"].tolist()

        for i, item in enumerate(row_names):
            # remove utterances without unique ids
            if f"{item}.wav" != "nan.wav":
                extract_portions_of_mp4_or_wav(
                    path_to_sound_file,
                    sound_file,
                    start_times[i],
                    end_timestamp[i],
                    save_path=f"{path_to_sound_file}/../split",
                    short_file_name=f"{item}.wav",
                )


def split_wavs_into_utterances(path_to_sound_files, df_with_timestamps):
    # split multiple wav files into single utterance files
    # get the set of all trials used here
    all_trials = set(df_with_timestamps["trial_id"].tolist())

    # set the path to the sound files
    filespath = Path(path_to_sound_files)
    # iterate over files in the sound file path to find wav files
    for item in filespath.iterdir():
        if item.suffix == ".wav":
            # get the trial and speaker for this file
            wav_trial = str(item.name).split("/")[-1].split("_")[2].split("-")[1]
            speaker = str(item.name).split("/")[-1].split("_")[4].split("-")[1]
            # if it is a trial of interest, split it into utterances
            if wav_trial in all_trials:
                short_df = df_with_timestamps[
                    df_with_timestamps["trial_id"] == wav_trial
                ]
                short_df = short_df[short_df["participant"] == speaker]
                split_wav_into_utterances(item.parents[0], item.name, short_df)


def convert_timestring_to_seconds(timestring):
    # convert a string representation of time to seconds
    # may be mm:ss, m:ss, mm:s, mm:ss:--
    time_sep = re.split(':|"', timestring)
    time_sep = [item for item in time_sep if item]  # removes empty strings
    if len(time_sep) == 1:
        return float(time_sep[0])
    elif len(time_sep) > 2:
        time_sep = time_sep[:2]
    return float(time_sep[0]) * 60 + float(time_sep[1])


if __name__ == "__main__":
    base_path = "/media/jculnan/One Touch/jculnan/datasets/MultiCAT"
    wav_path = f"{base_path}/individual_speaker_audio"
    df = pd.read_csv(f"{base_path}/processed_dataset_updated.csv")

    split_wavs_into_utterances(wav_path, df)
