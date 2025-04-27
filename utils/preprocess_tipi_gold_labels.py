# preprocess_tipi_gold-labels

import pandas as pd


def preprocess_tipi(df):
    """
    Given a pandas df containing the tipi data,
    preprocess this to calculate personality scores
    and find max personality trait for each person
    """
    # drop first two rows, as these are not data
    df = df.iloc[2:]

    # get only relevant columns
    used_cols = [
        "participantid",
        "TIPI_1",
        "TIPI_2",
        "TIPI_3",
        "TIPI_4",
        "TIPI_5",
        "TIPI_6",
        "TIPI_7",
        "TIPI_8",
        "TIPI_9",
        "TIPI_10",
    ]
    df.drop(columns=df.columns.difference(used_cols), inplace=True)

    # remove missing values
    df = df.dropna()

    df[
        [
            "TIPI_1",
            "TIPI_2",
            "TIPI_3",
            "TIPI_4",
            "TIPI_5",
            "TIPI_6",
            "TIPI_7",
            "TIPI_8",
            "TIPI_9",
            "TIPI_10",
        ]
    ] = df[
        [
            "TIPI_1",
            "TIPI_2",
            "TIPI_3",
            "TIPI_4",
            "TIPI_5",
            "TIPI_6",
            "TIPI_7",
            "TIPI_8",
            "TIPI_9",
            "TIPI_10",
        ]
    ].apply(
        pd.to_numeric
    )

    # calculate each trait score from tipi
    # each item is a 7 point likert scale
    # TIPI_1 = extroverted, enthusiastic          E
    # TIPI_2 = critical, quarrelsome              A - R
    # TIPI_3 = dependable, self-disciplined       C
    # TIPI_4 = anxious, easily upset              N
    # TIPI_5 = open to new experiences, complex   O
    # TIPI_6 = reserved, quiet                    E - R
    # TIPI_7 = sympathetic, warm                  A
    # TIPI_8 = disorganized, careless             C - R
    # TIPI_9 = calm, emotionally stable           N - R
    # TIPI_10 = conventional, uncreative          O - R
    # for each pair, one is flipped, so use item + (8 - item-r)
    df["extroversion"] = (df["TIPI_1"] + 8 - df["TIPI_6"]) / 2
    df["agreeableness"] = (8 - df["TIPI_2"] + df["TIPI_7"]) / 2
    df["conscientiousness"] = (df["TIPI_3"] + 8 - df["TIPI_8"]) / 2
    df["neuroticism"] = (df["TIPI_4"] + 8 - df["TIPI_9"]) / 2
    df["openness"] = (df["TIPI_5"] + 8 - df["TIPI_10"]) / 2

    df["max_trait"] = df[
        [
            "extroversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]
    ].idxmax(axis=1)

    return df


def add_traits_to_id_df(id_df, tipi_df):
    """
    Add personality trait scores to the df containing
    info on each participant (ID, role, team, trials)
    """
    tipi_df2 = tipi_df[
        [
            "participantid",
            "max_trait",
            "extroversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]
    ]
    id_df = id_df.merge(tipi_df2, on="participantid")

    return id_df
