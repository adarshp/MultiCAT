README
======

Code for the DA and AP classification results and the LLaMA3-based Sent/Emo
results for the MultiCAT paper (NAACL 2025 Findings).

Instructions
------------

Make sure to update the path to the database before running.

To generate data for LLaMa:

## Sent/Emo

    python preprocess_sentEmo.py
    python data_llama_sentEmo.py

## DA

    python preprocess_DA.py
    python data_llama_da.py

To generate data for He et al. (2021)

    python create_csvs.py
