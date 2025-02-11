-- Get all trials with aggregate label counts
WITH all_trials AS (
    SELECT
        trial.id AS trial,
        trial.score AS score,
        trial.team AS team,
        mission,
        COUNT(clc_label) AS n_clc_labels,
        COUNT(sentiment) AS n_sent_labels,
        COUNT(emotion) AS n_emo_labels,
        COUNT(dialog_acts) AS n_da_labels,
        COUNT(adjacency_pairs) AS n_ap_labels,
        COUNT(*) AS n_utts
    FROM
        trial
        JOIN utterance
    WHERE
        trial.id = utterance.trial
    GROUP BY
        trial
),
-- Select trials with all annotation types of interest
valid_trials AS (
    SELECT
        trial,
        team,
        mission,
        score,
        n_utts,
        n_clc_labels
    FROM
        all_trials
    WHERE
        n_clc_labels != 0
        AND n_sent_labels != 0
        AND n_emo_labels != 0
        AND n_da_labels != 0
        AND n_ap_labels != 0
),
-- Get aggregate participant info for each trial
valid_trials_with_participant_info AS (
    SELECT
        trial,
        mission,
        score,
        n_utts,
        n_clc_labels,
        -- Please rate how confident you are that you can learn and succeed at
        -- a new video game or set of game-related skills after minimal
        -- practice? - Confidence positivity (0--100)
        GROUP_CONCAT(participant.mc_prof_2_1) AS mc_prof_2_1,
        -- Learning layout of new virtual environment (0--100)
        GROUP_CONCAT(participant.mc_prof_4_1) AS mc_prof_4_1,
        -- Communicating your current location in a virtual environment to
        -- members of a team (0--100)
        GROUP_CONCAT(participant.mc_prof_4_2) AS mc_prof_4_2,
        -- Coordinating with teammates to optimize tasks (0--100)
        GROUP_CONCAT(participant.mc_prof_4_3) AS mc_prof_4_3,
        -- Maintaining an awareness of game/task parameters (e.g., time limits,
        -- point goals, etc (0--100))
        GROUP_CONCAT(participant.mc_prof_4_4) AS mc_prof_4_4,
        -- Learning the purposes of novel items, tools, or objects (0--100)
        GROUP_CONCAT(participant.mc_prof_4_5) AS mc_prof_4_5,
        -- Remembering which places you have visited  in a virtual environment (0--100)
        GROUP_CONCAT(participant.mc_prof_4_6) AS mc_prof_4_6,
        -- Controlling the movement of an avatar using the W, A, S, and D keys
        -- + mouse control (0--100)
        GROUP_CONCAT(participant.mc_prof_4_7) AS mc_prof_4_7,
        -- Keeping track of where you are in a virtual environment (0--100)
        GROUP_CONCAT(participant.mc_prof_4_8) AS mc_prof_4_8,
        -- Years using a computer for any purpose:
        GROUP_CONCAT(participant.mc_prof_9_1) AS mc_prof_9_1,
        -- Years using a computer to play video games:
        GROUP_CONCAT(participant.mc_prof_9_2) AS mc_prof_9_2,
        -- Years using a system other than a computer to play video games (e.g., mobile phone, gaming console, arcade console):
        GROUP_CONCAT(participant.mc_prof_9_3) AS mc_prof_9_3,
        -- Years playing Minecraft (any versions or styles of play):
        GROUP_CONCAT(participant.mc_prof_9_4) AS mc_prof_9_4
    FROM
        participant
        JOIN team ON participant.team = team.id
        JOIN valid_trials ON valid_trials.team = team.id
    GROUP BY
        trial
)
SELECT
    *
FROM
    valid_trials_with_participant_info
WHERE
    mission = 1
