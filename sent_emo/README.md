# Sent_Emo: Sentiment and Emotion Recognition

This directory contains code for reproducing the Stratified and Multitask baselines for sentiment/emotion classification tasks from the MultiCAT paper (NAACL Findings 2025).

## Overview

We introduce multiple approaches for sentiment and emotion recognition, ranging from lexicon-based methods to deep learning architectures. Our framework supports both unimodal (text-only) and multimodal analysis.

Please access our MultiCAT dataset via (`datasette_interface/multicat.db`) or Zenodo with DOI [10.5281/zenodo.14834836](https://doi.org/10.5281/zenodo.14834836), and then use the `data_preprocessing` module to extract and prepare the data for sentiment/emotion classification tasks.

## Repository Structure

```
sent_emo/
├── data_preprocessing/         # Data extraction and preprocessing utilities
├── emoroberta_text_only/       # EmoRoBERTa-based emotion recognition model
├── lexicon_based/              # NRC lexicon-based emotion analysis
├── multimodal/                 # Multimodal sentiment and emotion analysis
├── text_only/                  # Text-only sentiment and emotion models
├── utils/                      # Additional utilities
├── grid_config.py              # Grid search configuration
├── grid_search.py              # Grid search implementation
├── random_config.py            # Random search configuration
├── random_search.py            # Random search implementation
```