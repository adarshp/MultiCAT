# Sent_Emo: Sentiment and Emotion Recognition

This directory contains code for reproducing the Stratified and Multitask baselines for sentiment/emotion classification tasks from the MultiCAT paper (NAACL Findings 2025).

## Overview

We introduce multiple approaches for sentiment and emotion recognition, ranging from lexicon-based methods to deep learning architectures. Our framework supports both unimodal (text-only) and multimodal analysis.

For information about dataset access and preparation, please refer to the main repository README.

## Repository Structure

```
sent_emo/
├── data_preprocessing/         
├── emoroberta_text_only/       # EmoRoBERTa-based emotion recognition model
├── lexicon_based/              # NRC lexicon-based emotion analysis
├── multimodal/                 # Multimodal sentiment and emotion analysis
├── text_only/                  # Text-only sentiment and emotion models
├── utils/                      
├── grid_config.py              
├── grid_search.py              
├── random_config.py            
├── random_search.py            
```

## How to Run

### 1. Data Preprocessing

First, extract and preprocess the data from the MultiCAT database:

```bash
python data_preprocessing/multicat_db_preprocessing.py
```

This script extracts sentiment and emotion data from the MultiCAT database and creates the necessary preprocessing files for model training. The preprocessed data will be saved in the `data/` directory with the following structure:

```
data/
├── text_data/             # Preprocessed text features
│   ├── train.pickle
│   ├── dev.pickle
│   └── test.pickle
└── ys_data/               # Sentiment and emotion labels
    ├── train.pickle
    ├── dev.pickle
    └── test.pickle
```

### 2. Model Training

To train the text-only sentiment and emotion recognition model:

```bash
python text_only/text_only_train.py
```

This script trains the model according to configurations in `text_only_config.py` and saves results to the `output/text_only/` directory.

#### Model Outputs

After training, the output will be saved in the `output/text_only/` directory with the following structure:

```
output/text_only/
└── [experiment_id]_[description]_[date]/
    └── LR[lr]_BATCH[batch]_INT-OUTPUT[dim]_DROPOUT[val]_FC-FINALDIM[dim]/
        ├── [model_name].pt       # Saved model weights
        ├── config.py             # Copy of config used for training
        ├── loss.png              # Training/validation loss curve
        ├── macro-f1_task-0.png   # Sentiment F1 curve
        ├── macro-f1_task-1.png   # Emotion F1 curve
        └── log                   
```
        
### 3. Hyperparameter Tuning

For hyperparameter optimization:

```bash
python grid_search.py   # Grid search
python random_search.py # Random search
```

Configuration files: `grid_config.py` and `random_config.py`. Results are saved to the `output/` directory, including a CSV file with performance metrics for each parameter combination tested.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{pyarelal-etal-2025-multicat,
    title = "{M}ulti{CAT}: Multimodal Communication Annotations for Teams",
    author = "Pyarelal, Adarsh  and
      Culnan, John M  and
      Qamar, Ayesha  and
      Krishnaswamy, Meghavarshini  and
      Wang, Yuwei  and
      Jeong, Cheonkam  and
      Chen, Chen  and
      Miah, Md Messal Monem  and
      Hormozi, Shahriar  and
      Tong, Jonathan  and
      Huang, Ruihong",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.61/",
    pages = "1077--1111",
    ISBN = "979-8-89176-195-7",
    abstract = "Successful teamwork requires team members to understand each other and communicate effectively, managing multiple linguistic and paralinguistic tasks at once. Because of the potential for interrelatedness of these tasks, it is important to have the ability to make multiple types of predictions on the same dataset. Here, we introduce Multimodal Communication Annotations for Teams (MultiCAT), a speech- and text-based dataset consisting of audio recordings, automated and hand-corrected transcriptions. MultiCAT builds upon data from teams working collaboratively to save victims in a simulated search and rescue mission, and consists of annotations and benchmark results for the following tasks: (1) dialog act classification, (2) adjacency pair detection, (3) sentiment and emotion recognition, (4) closed-loop communication detection, and (5) vocal (phonetic) entrainment detection. We also present exploratory analyses on the relationship between our annotations and team outcomes. We posit that additional work on these tasks and their intersection will further improve understanding of team communication and its relation to team performance. Code {\&} data: https://doi.org/10.5281/zenodo.14834835"
}
```
