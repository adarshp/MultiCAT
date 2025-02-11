#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "tqdm",
#     "Jinja2"
# ]
# ///

import sqlite3
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Optional

pd.options.mode.copy_on_write = True
import matplotlib
from matplotlib import pyplot as plt

plt.style.use("ggplot")
from scipy.stats import sem, chi2_contingency
from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    SGDRegressor,
    ElasticNet,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


# Change the line below to reflect the actual location of the multicat SQLite3
# database, if it is not in the current working directory.
DB_PATH = "multicat.db"

DB_CONNECTION = sqlite3.connect(DB_PATH)
FEATURE_SETS = {
    "Proficiency": ["proficiency"],
    "AP": ["ap"],
    "CLC": ["clc"],
    "DA": ["da"],
    "Sent": ["sent"],
    "Emo": ["emo"],
    "Multicat": ["sent", "emo", "ap", "da", "clc"],
    "All": ["sent", "emo", "ap", "da", "clc", "proficiency"],
}

# Whether to group general DA tags
GROUP_DA_TAGS = True
GROUP_CLC_TAGS = True

rng = np.random.RandomState(0)


def convert_da(da_label_with_pipes):
    """
    An utterance can have multiple DA labels separated by pipes (cases where
    segments of the utterance require different tags and cannot be merged into
    one label because of different general tags.)

    Each DA label is of the form <gen>^<spec>^<spec>^...

    where <gen> is a general DA tag, and <spec> is a specific DA tag.

    See the MRDA manual for details.
    """
    #  Get the list of DA labels for an utterance (splitting on |)
    da_labels = str(da_label_with_pipes).split("|")
    all_general_tags = []
    all_specific_tags = []
    incomplete = False
    for label in da_labels:
        general_tag, specific_tags, includes_incomplete = convert_single_da(
            label.strip()
        )
        if general_tag is not None:
            all_general_tags.append(general_tag)
        if specific_tags is not None:
            all_specific_tags.append(specific_tags)
        if includes_incomplete:
            incomplete = True

    # convert to set to remove duplicates
    all_specific_tags = sorted(list(set(all_specific_tags)))
    all_general_tags = sorted(list(set(all_general_tags)))
    if len(all_specific_tags) == 0:
        all_specific_tags = ["None"]

    general_tag = None

    if GROUP_DA_TAGS:
        if len(all_general_tags) == 1:
            general_tag = all_general_tags[0]
            if general_tag in {"qy", "qw", "qr", "qrr", "qo", "qh"}:
                general_tag = "question"
            elif general_tag in {"fg", "fh", "h"}:
                general_tag = "floor_mechanism"
            elif general_tag in {"b", "bk", "ba", "bh"}:
                general_tag = "backchannels_acknowledgments"
            else:
                pass
        elif len(all_general_tags) > 1:
            general_tag = "multiple"
        else:
            pass
    else:
        if len(all_general_tags) == 0:
            general_tag = "None"
        if len(all_general_tags) == 1:
            general_tag = all_general_tags[0]
        else:
            general_tag = "_".join(all_general_tags)

    return general_tag, "_".join(all_specific_tags), incomplete


def convert_single_da(label) -> (Optional[str], Optional[str], bool):
    """
    Convert a single label.
    """
    if label == "":
        return None, None, False
    else:
        label, includes_incomplete = remove_incomplete(label)
        tags = label.split("^")
        tags = [tag for tag in tags if tag]
        general_tag = None
        specific_tags = None

        if len(tags) == 1:
            general_tag = tags[0]
        elif len(tags) > 1:
            specific_tags = "_".join(tags[1:])
        else:
            pass

        return general_tag, specific_tags, includes_incomplete


def remove_incomplete(label) -> (str, bool):
    """Returns a version of the label with incomplete tags removed, and whether
    there were any incomplete tags in the label"""
    if ".%--" in label:
        item = "".join(label.split(".%--"))
        return item, True
    elif ".%-" in label:
        item = "".join(label.split(".%-"))
        return item, True
    elif ".%" in label:
        item = "".join(label.split(".%"))
        return item, True
    elif "%" in label:
        item = "".join(label.split("%"))
        return item, True
    else:
        return label, False


# convert items in AP, if needed
def convert_ap(ap_label):
    """
    Convert AP labels
    Just get A & B
    AP labels of format 1B+.2A
    """
    # split on period
    ap_label = str(ap_label)
    ap_labels = ap_label.split(".")

    types = []
    # ID A & B from within this
    for label in ap_labels:
        if label.lower() == "nan":
            continue
        elif "b" in label.lower():
            types.append("b")
        elif "a" in label.lower():
            types.append("a")

    # if "a" only, return "a"
    # if "b" only, return "b"
    # if neither, return "ap_neither"
    # if both, return "ap_both"
    types = sorted(list(set(types)))
    if len(types) == 0:
        return "ap_neither"
    elif len(types) > 1:
        return "ap_both"
    else:
        return types[0]


# convert items in CLC, if needed
def convert_clc(clc_label):
    """
    Convert CLC labels
    Just get A, B, & C
    CLC labels of format 1B+.2A
    """
    # split on period
    clc_label = str(clc_label)
    clc_labels = clc_label.split(".")

    types = []
    # ID A & B from within this
    for label in clc_labels:
        if label.lower() == "nan":
            continue
        elif "b" in label.lower():
            types.append("b")
        elif "a" in label.lower():
            types.append("a")
        elif "c" in label.lower():
            types.append("c")

    types = list(set(types))
    if len(types) == 0:
        return "clc_none"
    else:
        if GROUP_CLC_TAGS:
            return "clc_some"
        else:
            if len(types) > 2:
                return "clc_all"
            elif len(types) == 2:
                if "a" in types and "b" in types:
                    return "clc_ab"
                elif "a" in types and "c" in types:
                    return "clc_ac"
                elif "b" in types and "c" in types:
                    return "clc_bc"
            else:
                return f"clc_{types[0]}"


def get_multiple_new_labels(df, da=True, ap=True, incomplete=True, clc=True):
    """Complete cleaning as required for the analysis of interest. Add columns
    to the dataframe corresponding to new label types (general/incomplete)."""

    if da:
        df = get_general_da_labels(df)
    if ap:
        df = get_general_ap_labels(df)
    if incomplete:
        df = get_incomplete_items(df)
    if clc:
        df = get_general_clc_labels(df)
    return df


def get_general_da_labels(df) -> pd.DataFrame:
    """Get 'general' DA tags from the more complex set"""
    df["general_da"] = df["dialog_acts"].apply(lambda x: convert_da(x)[0])
    return df


def get_incomplete_items(df):
    """Get incomplete items in a df"""
    df["incomplete_utt"] = df["dialog_acts"].apply(lambda x: convert_da(x)[2])
    return df


def get_general_ap_labels(df):
    """Get general AP labels from the more complex set"""
    df["ap_types"] = df["adjacency_pairs"].apply(convert_ap)
    return df


def get_general_clc_labels(df):
    """Get general CLC labels from specific ones"""
    df["clc"] = df["clc_label"].apply(convert_clc)
    return df


def get_preprocessed_data(condition, results_df):
    """Get the design matrix X and the outputs y"""

    # Get the trials that have all types of annotations.

    query_files = {
        "Mission 1": "query_mission_1_only.sql",
        "Mission 2": "query_mission_2_only.sql",
        "Both missions": "query_both_missions.sql",
    }

    with open(query_files[condition]) as f:
        query = f.read()

    df = pd.read_sql(query, DB_CONNECTION, index_col="trial")
    results_df.loc[condition, "N"] = len(df)

    mc_prof_suffixes = (
        ["2_1"]
        + [f"4_{n}" for n in range(1, 9)]
        + [f"9_{n}" for n in range(1, 5)]
    )
    for column in mc_prof_suffixes:
        df[f"avg_mc_prof_{column}"] = df[f"mc_prof_{column}"].apply(
            lambda xs: np.mean([int(x) for x in xs.split(",")])
        )

    for trial in df.index:
        labels = pd.read_sql(
            f"SELECT * FROM utterance WHERE trial = '{trial}'",
            DB_CONNECTION,
        )

        labels = get_multiple_new_labels(labels)
        labels["specific_da"] = labels["dialog_acts"].apply(
            lambda x: convert_da(x)[1]
        )

        clc_labels = labels["clc_label"].values
        sentiment_labels = labels["sentiment"].values
        n_utts = df.loc[trial, "n_utts"]

        # Get number of utterances with 'a', 'b', 'c' CLC labels
        n_a = len([x for x in clc_labels if x is not None and "a" in x])
        n_b = len([x for x in clc_labels if x is not None and "b" in x])
        n_c = len([x for x in clc_labels if x is not None and "c" in x])

        n_utts = df.loc[trial, "n_utts"]

        # Get fraction of closed loops
        f_c = n_c / n_a

        df.loc[trial, "f_c"] = f_c

        emos = labels.emotion.value_counts().to_dict()
        for emo in emos:
            new_emo = emo
            if emo == "neutral":
                new_emo = "emo_neutral"
            df.loc[trial, new_emo] = emos[emo]

        sents = labels.sentiment.value_counts().to_dict()
        for sent in sents:
            new_sent = sent
            if sent == "neutral":
                new_sent = "sent_neutral"
            df.loc[trial, new_sent] = sents[sent]

        aps = labels.ap_types.value_counts().to_dict()
        for ap in aps:
            df.loc[trial, ap] = aps[ap]

        # get summary of features for da
        das = labels.general_da.value_counts().to_dict()
        for da in das:
            df.loc[trial, da] = das[da]

        # get summary of features for clc
        clcs = labels.clc.value_counts().to_dict()
        for clc in clcs:
            df.loc[trial, clc] = clcs[clc]

        # Get number of positive utterances
        n_pos = (
            len(
                [
                    x
                    for x in sentiment_labels
                    if x is not None and "positive" in x
                ]
            )
            / n_utts
        )

        # Get fraction of negative utterances
        n_neg = (
            len(
                [
                    x
                    for x in sentiment_labels
                    if x is not None and "negative" in x
                ]
            )
            / n_utts
        )
        df.loc[trial, "n_a"] = n_a
        df.loc[trial, "n_b"] = n_b
        df.loc[trial, "n_c"] = n_c
        df.loc[trial, "n_pos"] = n_pos
        df.loc[trial, "n_neg"] = n_neg

    df.drop(
        [f"mc_prof_{suffix}" for suffix in mc_prof_suffixes],
        axis=1,
        inplace=True,
    )

    y = df["score"].values
    df = df.drop(
        ["score", "mission"],
        axis=1,
    )

    # Change NaN to 0
    df.fillna(0, inplace=True)
    X = df
    return X, y


def plot_data():
    # Plotting
    X, y = get_preprocessed_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    for feature in X.columns:
        fig, ax = plt.subplots()
        ax.scatter(X[feature].values, y)
        ax.set_xlabel(feature)


def evaluate_score_prediction_model(X, y, results_df, condition):
    """Run experiment with preprocessed data, including cross-validation."""

    cv = KFold(n_splits=len(y), shuffle=True)

    all_features = {
        "proficiency": [
            "avg_mc_prof_2_1",
            "avg_mc_prof_4_1",
            "avg_mc_prof_4_2",
            "avg_mc_prof_4_3",
            "avg_mc_prof_4_4",
            "avg_mc_prof_4_5",
            "avg_mc_prof_4_6",
            "avg_mc_prof_4_7",
            "avg_mc_prof_4_8",
            "avg_mc_prof_9_1",
            "avg_mc_prof_9_2",
            "avg_mc_prof_9_3",
            "avg_mc_prof_9_4",
        ],
        "emo": [
            "emo_neutral",
            "joy",
            "surprise",
            "sadness",
            "disgust",
            "anger",
            "fear",
        ],
        "sent": ["sent_neutral", "positive", "negative"],
        "ap": ["ap_neither", "b", "a", "ap_both"],
        "other": [
            "n_utts",
            "n_clc_labels",
            "n_a",
            "n_b",
            "n_c",
            "f_c",
        ],
        "n_utts": ["n_utts"],
    }

    all_features["clc"] = (
        [
            "clc_none",
            "clc_some",
        ]
        if GROUP_CLC_TAGS
        else [
            "clc_none",
            "clc_a",
            "clc_b",
            "clc_ab",
            "clc_c",
            "clc_ac",
            "clc_bc",
            # "f_c"
        ]
    )

    all_features["da"] = (
        [
            "multiple",
            "question",
            "backchannels_acknowledgments",
            "s",
            "x",
        ]
        if GROUP_DA_TAGS
        else [
            "s",
            "qy",
            "qy_s",
            "qw",
            "qw_s",
            "qw_qy",
            "qr",
            "qw_qy_s",
            "x",
            "qr_s",
            "qr_qy_s",
            "z",
            "qo_s",
        ]
    )

    with open("../../../sections/score_prediction_results.tex", "w") as f:
        maes = []
        maes_mean_baseline = []
        # Report results for different models

        # Loop over models
        for set_name, feature_collection in FEATURE_SETS.items():
            features = []
            for collection in feature_collection:
                features.extend(all_features[collection])

            print(set_name, features)

            print(f"{len(features)=}")

            for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X, y)),
                total=cv.get_n_splits(),
                unit="folds",
            ):
                X_subset = X[features]
                X_train = X_subset.values[train_idx]
                y_train = y[train_idx]

                # Mean baseline model
                y_train_mean = y_train.mean()

                X_test = X_subset.values[test_idx]
                y_test = y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                n_components = 1
                pca = PCA(n_components)
                X_train_scaled_pca = pca.fit_transform(X_train_scaled)

                # Initialize the model
                model = Ridge(10)

                # Get predictions
                model.fit(X_train_scaled_pca, y_train)
                y_pred = model.predict(pca.transform(scaler.transform(X_test)))
                mae = median_absolute_error(y_test, y_pred)
                maes.append(mae)
                mae_mean_baseline = np.median(
                    np.abs(y_test - y_train_mean * np.ones(len(y_test)))
                )
                maes_mean_baseline.append(mae_mean_baseline)

            mean_mae = np.mean(maes)

            # Compute standard error of the mean
            sem_mae = sem(maes)
            results_df.loc[
                condition, set_name
            ] = f"{mean_mae:.0f} ({sem_mae:.0f})"

            mean_mae_mean_baseline = np.mean(maes_mean_baseline)

            # Compute standard error of the mean
            sem_mae_mean_baseline = sem(maes_mean_baseline)

        f.write(results_df.T.to_latex())


def cat2_relationship(df, label_1, label_2):
    """
    Get relationship between two categorical variables
    """
    crosstab = pd.crosstab(df[label_1], df[label_2])
    crosstab.index = [i.replace("_", "\\_") for i in crosstab.index]
    crosstab.columns = [i.replace("_", "\\_") for i in crosstab.columns]
    with open(
        f"../../../appendices/crosstab_{label_1}_{label_2}.tex", "w"
    ) as f:
        f.write(crosstab.to_latex())

    chi2_res = chi2_contingency(crosstab, lambda_="log-likelihood")
    flattened_expected_frequency_table = chi2_res.expected_freq.flatten()

    # Check number of entries less than 5
    f_lt_5 = len(
        [x for x in flattened_expected_frequency_table if x < 5]
    ) / len(flattened_expected_frequency_table)

    print(f"No more than 20% of entries < 5?: {f_lt_5 < .2}")

    all_expected_counts_gt_1 = all(
        [True if x > 1 else False for x in flattened_expected_frequency_table]
    )
    # Check if all expected counts are > 1
    print(f"All expected counts > 1?: {all_expected_counts_gt_1}")
    print(
        f"$\\chi^2({chi2_res.dof}) = {chi2_res.statistic:.2f}, p = {chi2_res.pvalue}"
    )
    print("")

    pvalue = (
        chi2_res.pvalue
        if (f_lt_5 < 0.2 and all_expected_counts_gt_1)
        else None
    )
    return crosstab, pvalue


def get_crosstabs():
    with DB_CONNECTION:
        dset_df = pd.read_sql_query("SELECT * FROM utterance", DB_CONNECTION)

    # convert items
    dset_df["general_da"] = dset_df["dialog_acts"].apply(
        lambda x: convert_da(x)[0]
    )
    dset_df["specific_da"] = dset_df["dialog_acts"].apply(
        lambda x: convert_da(x)[1]
    )
    dset_df["incomplete_utt"] = dset_df["dialog_acts"].apply(
        lambda x: convert_da(x)[2]
    )

    dset_df["ap_types"] = dset_df["adjacency_pairs"].apply(
        lambda x: convert_ap(x)
    )
    dset_df["clc"] = dset_df["clc_label"].apply(lambda x: convert_clc(x))

    p_values = {}

    for combination in combinations(
        ["clc", "sentiment", "ap_types", "general_da", "emotion"], 2
    ):
        print(f"Combination: {combination}")
        crosstab, pvalue = cat2_relationship(
            dset_df, combination[0], combination[1]
        )
        p_values[combination] = pvalue

    n_tests = len([x for x in p_values.values() if x is not None])
    print(n_tests)

    bonferroni_critical_value = 0.001 / n_tests
    print(bonferroni_critical_value)

    for k, v in p_values.items():
        print(f"{k}: {v}")
        if v is not None:
            print(
                f"Less than Bonferroni critical value? {v < bonferroni_critical_value}"
            )


def run_score_prediction_experiment():
    conditions = ["Mission 1", "Mission 2", "Both missions"]
    results_df = pd.DataFrame(
        index=conditions,
        columns=["N"] + list(FEATURE_SETS.keys()),
    )
    for condition in conditions:
        X, y = get_preprocessed_data(condition, results_df)
        evaluate_score_prediction_model(X, y, results_df, condition)


if __name__ == "__main__":
    run_score_prediction_experiment()
    get_crosstabs()
