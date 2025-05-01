# use this script to extract acoustic features from your datasets
# this script assumes that all of the datasets you will use
#   exist in the same base directory

import os
from tqdm import tqdm
import subprocess as sp


class ExtractAudio:
    """
    Takes audio and extracts features from it using openSMILE
    """

    def __init__(self, path, audiofile, savedir, smilepath="~/opensmile-3.0"):
        self.path = path
        self.afile = path + "/" + audiofile
        self.savedir = savedir
        self.smile = smilepath

    def save_acoustic_csv(self, feature_set, savename):
        """
        Get the CSV for set of acoustic features for a .wav file
        feature_set : the feature set to be used
        savename : the name of the saved CSV
        Saves the CSV file
        """
        conf_dict = {
            "ISO9": "is09-13/IS09_emotion.conf",
            "IS10": "is09-13/IS10_paraling.conf",
            "IS12": "is09-13/IS12_speaker_trait.conf",
            "IS13": "is09-13/IS13_ComParE.conf",
        }

        fconf = conf_dict.get(feature_set, "IS13_ComParE.conf")

        # check to see if save path exists; if not, make it
        os.makedirs(self.savedir, exist_ok=True)

        # run openSMILE
        sp.run(
            [
                f"{self.smile}/build/progsrc/smilextract/SMILExtract",
                "-C",
                f"{self.smile}/config/{fconf}",
                "-I",
                self.afile,
                "-lldcsvoutput",
                f"{self.savedir}/{savename}",
            ]
        )


def run_feature_extraction(audio_path, feature_set, save_dir):
    """
    Run feature extraction from audio_extraction.py for a dataset
    :param audio_path: the full path to the directory containing audio files
    :param feature_set: the feature set used;
        For openSMILE, the feature set is one of IS09-IS13
        For spectrograms, feature set is `spectrogram` or `spec`
        For ASR-preprocessed feats, feature set is `asr` or `wav2vec`
    """
    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_dir}" ]; then mkdir -p {save_dir}; fi')

    # save all files in the directory
    for wfile in tqdm(os.listdir(audio_path), desc=f"Processing files in {audio_path}"):
        if wfile.endswith(".wav"):
            save_name = str(wfile.split(".wav")[0]) + f"_{feature_set}.csv"
            if feature_set.lower() in ["is09", "is10", "is11", "is12", "is13"]:
                audio_extractor = ExtractAudio(
                    audio_path, wfile, save_dir, "../../opensmile-3.0"
                )
                audio_extractor.save_acoustic_csv(feature_set, save_name)


if __name__ == "__main__":
    base_path = "/media/jculnan/One Touch/jculnan/datasets/MultiCAT"
    run_feature_extraction(f"{base_path}/split", "IS13", f"{base_path}/IS13")
