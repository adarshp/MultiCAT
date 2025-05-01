from sklearn.metrics import cohen_kappa_score
import pandas as pd


class IRR:
    def __init__(self, dataframe1_paths, dataframe2_paths):
        """
        Initialize an IRR object
        :param dataframe1_paths: A list of filepaths for annotator 1
        :param dataframe2_paths: A list of filepaths for annotator 2
        """
        print(dataframe1_paths)
        print(dataframe2_paths)
        self.df1 = self._concat_files(dataframe1_paths)
        self.df2 = self._concat_files(dataframe2_paths)

        self.valid_args = {
            "sentiment": ["negative", "neutral", "positive"],
            "emotion": [
                "anger",
                "disgust",
                "fear",
                "joy",
                "neutral",
                "sadness",
                "surprise",
            ],
            "addressee": ["all:", "engineer:", "medic:", "transporter:"],
        }

    def save_combined(self, d1_savename, d2_savename):
        if d1_savename:
            self.df1.to_csv(d1_savename, index=False)
        if d2_savename:
            self.df2.to_csv(d2_savename, index=False)

    def get_data_distribution(self, annotation_type):
        """
        Get the distribution of data across classes for a given annotation type
        :param annotation_type: a string with the name of the annotation type
            'sentiment' for sentiment annotation
            'emotion' for emotion annotation
        """
        return (
            self.df1[annotation_type].value_counts(),
            self.df2[annotation_type].value_counts(),
        )

    def get_kappas(self, annotation_type, by_category=True):
        """
        Get cohen's kappa scores for sentiment and emotion annotations
        :param annotation_type: a string with the name of the annotation type
            'sentiment' for sentiment annotation
            'emotion' for emotion annotation
        :param by_category: a boolean;
            true: get the kappa for each sent/emotion category
            false: get only the overall kappa
        :returns the cohen's kappa scores and value counts for the annotation
            type in each df
        """
        if annotation_type in self.valid_args.keys():
            valid = self.valid_args[annotation_type]
            df1 = self.df1
            df2 = self.df2
        else:
            df1 = self.df1
            df2 = self.df2

        df1[annotation_type] = df1[annotation_type].apply(lambda x: quick_data_cleanup(x))
        df2[annotation_type] = df2[annotation_type].apply(lambda x: quick_data_cleanup(x))

        # drop all items where df1 AND df2 value == 'none'
        # to only capture items where one person provided annotation
        # and the other did not
        anns_1 = df1[annotation_type].tolist()
        anns_2 = df2[annotation_type].tolist()

        final_anns_1 = []
        final_anns_2 = []
        for i, item in enumerate(anns_1):
            if item == 'none' and anns_2[i] == 'none':
                continue
            else:
                final_anns_1.append(item)
                final_anns_2.append(anns_2[i])

        if by_category:
            kappa = {}
            for item in valid:
                anns_1_cat = []
                anns_2_cat = []

                for i, wd in enumerate(final_anns_1):
                    if wd == item:
                        anns_1_cat.append(wd)
                        anns_2_cat.append(final_anns_2[i])
                    elif final_anns_2[i] == item:
                        anns_1_cat.append(wd)
                        anns_2_cat.append(final_anns_2[i])

                # use anns_1 as gold and anns_2 as pred
                true_neg = len(final_anns_2) - len(anns_2_cat)
                true_pos = len([w for idx, w in enumerate(anns_2_cat) if w == anns_1_cat[idx]])
                false_neg = len([w for idx, w in enumerate(anns_2_cat) if anns_2_cat[idx] != item])
                false_pos = len([w for idx, w in enumerate(anns_2_cat) if anns_1_cat[idx] != item])

                # calculate po, p1, p2, pe
                po = (true_pos + true_neg) / len(final_anns_2)
                p1 = (true_pos + false_neg) * (true_pos + false_pos) / (len(final_anns_2) ** 2)
                p2 = (true_neg + false_neg) * (true_neg + false_pos) / (len(final_anns_2) ** 2)
                pe = p1 + p2

                k = (po - pe) / (1 - pe)

                kappa[item] = k

            kappa['overall'] = cohen_kappa_score(final_anns_1, final_anns_2)
            return kappa
        else:
            kappa = cohen_kappa_score(final_anns_1, final_anns_2)
            return kappa

    def _concat_files(self, list_of_files):
        """
        Concatenate a list of files
        :return:
        """
        all_files = None
        for file in list_of_files:
            pd_df = pd.read_csv(file)
            pd_df.columns = [colname.lower() for colname in pd_df.columns]

            if all_files is None:
                all_files = pd_df
            else:
                all_files = pd.concat([all_files, pd_df], axis=0)

        return all_files


def quick_data_cleanup(item):
    """
    Clean up input data to handle typos
    """
    if item in ['negtive', 'negatve']:
        return 'negative'
    elif str(item) == 'neural':
        return 'neutral'
    elif str(item) == "0" or type(item) == float:
        return 'none'
    elif item  == 'angry':
        return 'anger'
    elif item == 'suprise':
        return 'surprise'
    elif item == 'sadeness':
        return 'sadness'
    else:
        return item.strip()


if __name__ == "__main__":
    ann1 = [
        "~/datasets/MultiCAT/annotations/sent-emo/CKJ_HSRData_TrialMessages_Trial-T000604_Team-TM000202_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv",
        "~/datasets/MultiCAT/annotations/sent-emo/CKJ_HSRData_TrialMessages_Trial-T000605_Team-TM000203_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv",
        "~/datasets/MultiCAT/annotations/sent-emo/CKJ_HSRData_TrialMessages_Trial-T000607_Team-TM000204_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv",
        "~/datasets/MultiCAT/annotations/sent-emo/CKJ_HSRData_TrialMessages_Trial-T000628_Team-TM000214_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv"
    ]

    ann2 = [
        "~/datasets/MultiCAT/annotations/sent-emo/SH_HSRData_TrialMessages_Trial-T000604_Team-TM000202_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv",
        "~/datasets/MultiCAT/annotations/sent-emo/SH_HSRData_TrialMessages_Trial-T000605_Team-TM000203_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv",
        "~/datasets/MultiCAT/annotations/sent-emo/SH_HSRData_TrialMessages_Trial-T000607_Team-TM000204_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv",
        "~/datasets/MultiCAT/annotations/sent-emo/SH_HSRData_TrialMessages_Trial-T000628_Team-TM000214_Member-na_CondBtwn-none_CondWin-na_Vers-3_correctedTranscripts.csv"
    ]

    x = IRR(ann1, ann2)

    # x.save_combined(
    #     "~/datasets/MultiCAT/annotations/sent-emo/CKJ_for_irr.csv",
    #     "~/datasets/MultiCAT/annotations/sent-emo/SH_for_irr.csv"
    #     )

    print("Cohen's kappa for sentiment")
    print(x.get_kappas("sentiment", by_category=True))
    print("Cohen's kappa for emotion")
    print(x.get_kappas("emotion", by_category=True))
    print(x.get_data_distribution("sentiment"))
    print(x.get_data_distribution('emotion'))
