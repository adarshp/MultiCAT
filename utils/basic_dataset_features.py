# get summary statistics and basic details of dataset
import pandas as pd
import torchtext
import numpy as np


def get_speakers_teams_trials(dset):
    """
    Get the number of speakers, teams, and trials
    :param dset:
    :return:
    """
    num_speakers = len(dset["participant"].unique().tolist())
    num_teams = len(dset["team_id"].unique().tolist())
    num_trials = len(dset["trial_id"].unique().tolist())

    return num_speakers, num_teams, num_trials


def get_summary_statistics_on_utts(dset):
    """
    Get summary statistics for the utterances
    :param dset:
    :return:
    """
    all_utts = dset["utt"].tolist()

    # get number of utterances in dset
    num_utts = len(all_utts)

    # get nested list of tokenized utts
    # a list of their lengths and a list of their set lengths
    tokenized_utts, utt_lens, utt_set_lens = get_tokenized_utts(all_utts)

    # get word types
    wds = [wd for utt in tokenized_utts for wd in utt]

    # get average of word types, tokens per utt
    # word types
    wd_types = np.array(utt_set_lens)
    all_types = len(set(wds))
    mean_types = np.mean(wd_types)
    min_types = min(wd_types)
    max_types = max(wd_types)
    std_types = np.std(wd_types)

    # word tokens
    all_tokens = len(wds)
    utt_lens = np.array(utt_lens)
    mean_tokens = np.mean(utt_lens)
    min_tokens = min(utt_lens)
    max_tokens = max(utt_lens)
    std_tokens = np.std(utt_lens)

    return (
        num_utts,
        all_tokens,
        all_types,
        mean_tokens,
        min_tokens,
        max_tokens,
        std_tokens,
        mean_types,
        min_types,
        max_types,
        std_types,
    )


def get_statistics_by_mission(dset):
    all_missions = dset["trial_id"].unique().tolist()

    utts_per_mission = []
    utts_per_speaker = []

    utts_per_speaker_total = {}

    # get avg of utts per trial
    for mission in all_missions:
        mission_utts = 0
        # get mission df
        mission_df = dset[dset["trial_id"] == mission]
        # get list of speakers
        speakers = mission_df["participant"].unique().tolist()

        # for speaker in mission
        for speaker in speakers:
            # get stats for that speaker
            n_utts = get_num_utts_for_one_speaker(mission_df, speaker)
            mission_utts += n_utts
            utts_per_speaker.append(n_utts)
            if speaker not in utts_per_speaker_total.keys():
                utts_per_speaker_total[speaker] = n_utts
            else:
                utts_per_speaker_total[speaker] += n_utts

        utts_per_mission.append(mission_utts)

    # get avg of utts per speaker in a trial
    # calculate avg, min, max, std per speaker in trial
    total_per_speaker = np.array([item for item in utts_per_speaker_total.values()])
    mean_per_speaker = np.mean(total_per_speaker)
    min_per_speaker = min(total_per_speaker)
    max_per_speaker = max(total_per_speaker)
    std_per_speaker = np.std(total_per_speaker)

    # calc avg, min, max, std per speaker
    utts_per_speaker = np.array(utts_per_speaker)
    mean_per_speaker_in_trial = np.mean(utts_per_speaker)
    min_per_speaker_in_trial = min(utts_per_speaker)
    max_per_speaker_in_trial = max(utts_per_speaker)
    std_per_speaker_in_trial = np.std(utts_per_speaker)

    # calc avg, min, max, std, per trial
    utts_per_mission = np.array(utts_per_mission)
    mean_per_trial = np.mean(utts_per_mission)
    min_per_trial = min(utts_per_mission)
    max_per_trial = max(utts_per_mission)
    std_per_trial = np.std(utts_per_mission)

    # return these
    return (
        mean_per_speaker,
        min_per_speaker,
        max_per_speaker,
        std_per_speaker,
        mean_per_speaker_in_trial,
        min_per_speaker_in_trial,
        max_per_speaker_in_trial,
        std_per_speaker_in_trial,
        mean_per_trial,
        min_per_trial,
        max_per_trial,
        std_per_trial,
    )


def get_num_utts_for_one_speaker(dset, speaker):
    """
    Get summary statistics for a single mission
    :param dset:
    :param speaker:
    :return:
    """
    # speaker portion
    speaker_df = dset[dset["participant"] == speaker]

    all_utts = speaker_df["utt"].tolist()

    # get number of utterances in dset
    num_utts = len(all_utts)

    return num_utts


def get_tokenized_utts(utts_list):
    """
    Get a list of tokenized utts from a list of utts
    :param utts_list:
    :return:
    """
    # get number of wd types, tokens in dset
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    tokenized_utts = []
    utt_lens = []
    utt_set_lens = []
    for utt in utts_list:
        tokenized_utt = tokenizer(utt)
        # append
        tokenized_utts.append(tokenized_utt)
        utt_lens.append(len(tokenized_utt))
        utt_set_lens.append(len(set(tokenized_utt)))

    return tokenized_utts, utt_lens, utt_set_lens


def speakers_teams_print_wrapper(dset):
    speakers, teams, trials = get_speakers_teams_trials(dset)

    print("**********************************")
    print("NOW PRINTING SPEAKERS, TEAMS, AND TRIALS")
    print("==================================")
    print("==================================")
    print(f"SPEAKERS:\t{speakers}")
    print(f"TEAMS:\t\t{teams}")
    print(f"TRIALS:\t\t{trials}")
    print("==================================")


def per_speaker_stats_print_wrapper(dset):
    (
        mean_per_speaker,
        min_per_speaker,
        max_per_speaker,
        std_per_speaker,
        mean_per_speaker_in_trial,
        min_per_speaker_in_trial,
        max_per_speaker_in_trial,
        std_per_speaker_in_trial,
        mean_per_trial,
        min_per_trial,
        max_per_trial,
        std_per_trial,
    ) = get_statistics_by_mission(dset)

    print("**********************************")
    print("PER SPEAKER STATS")
    print("==================================")
    print("==================================")
    print(f"MEAN:\t\t{mean_per_speaker}")
    print(f"MIN:\t\t{min_per_speaker}")
    print(f"MAX:\t\t{max_per_speaker}")
    print(f"STD:\t\t{std_per_speaker}")
    print("**********************************")
    print("PER TRIAL STATS")
    print("==================================")
    print("==================================")
    print(f"MEAN:\t\t{mean_per_trial}")
    print(f"MIN:\t\t{min_per_trial}")
    print(f"MAX:\t\t{max_per_trial}")
    print(f"STD:\t\t{std_per_trial}")
    print("**********************************")
    print("PER SPEAKER IN ONE TRIAL STATS")
    print("==================================")
    print("==================================")
    print(f"MEAN:\t\t{mean_per_speaker_in_trial}")
    print(f"MIN:\t\t{min_per_speaker_in_trial}")
    print(f"MAX:\t\t{max_per_speaker_in_trial}")
    print(f"STD:\t\t{std_per_speaker_in_trial}")


def summary_stats_print_wrapper(dset):
    # print summary stats
    (
        num_utts,
        all_tokens,
        all_types,
        mean_tokens,
        min_tokens,
        max_tokens,
        std_tokens,
        mean_types,
        min_types,
        max_types,
        std_types,
    ) = get_summary_statistics_on_utts(dset)

    print("**********************************")
    print("OVERALL STATS")
    print("==================================")
    print("==================================")
    print(f"NUM UTTS:\t{num_utts}")
    print(f"NUM TOKENS:\t{all_tokens}")
    print(f"MEAN TOKENS:\t{mean_tokens}")
    print(f"MIN TOKENS:\t{min_tokens}")
    print(f"MAX TOKENS:\t{max_tokens}")
    print(f"STD TOKENS:\t{std_tokens}")
    print("==================================")
    print(f"NUM TYPES:\t{all_types}")
    print(f"MEAN TYPES:\t{mean_types}")
    print(f"MIN TYPES:\t{min_types}")
    print(f"MAX TYPES:\t{max_types}")
    print(f"STD TYPES:\t{std_types}")
    print("==================================")


if __name__ == "__main__":
    dset_path = "/media/jculnan/One Touch/jculnan/datasets/MultiCAT/processed_dataset_updated.csv"
    # dset_path = "/media/jculnan/One Touch/jculnan/datasets/MultiCAT/da_sentemo_clc.csv"
    dset = pd.read_csv(dset_path)

    # remove items without an utt
    dset = dset.dropna(subset=["utt"])

    # get number of speakers, teams, and trials
    speakers_teams_print_wrapper(dset)

    # get summary statistics for the whole dataset
    summary_stats_print_wrapper(dset)

    # get per speaker/trial stats
    per_speaker_stats_print_wrapper(dset)
