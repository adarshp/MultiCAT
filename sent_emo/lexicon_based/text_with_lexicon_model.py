# baseline model for sentiment and emotion classification

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

from baselines.sent_emo.lexicon_based.exploring_NRC_emotion import load_nrc_emotion_as_dict, get_emo_vec
from baselines.sent_emo.text_only.text_only_model import RobertaBase, PredictionLayer


class EmoLexiconModelBase(nn.Module):
    """
    An encoder to take a sequence of inputs and
    produce a sequence of intermediate representations
    """

    def __init__(self, params, device, model_type="roberta-base"):
        super(EmoLexiconModelBase, self).__init__()
        # set roberta base
        self.text_roberta = RobertaBase(
            device=device, model_type=model_type, config=params
        )

        # set emotion lexicon
        self.nrc_emotion = load_nrc_emotion_as_dict(params.nrc_path)
        self.lemmatizer = WordNetLemmatizer()

        self.lexicon_len = 10  # 8 emotions + 2 sentiment classes from the lexicon

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize fully connected layers
        self.fc1 = nn.Linear(params.text_dim + self.lexicon_len, params.fc_hidden_dim)

    def forward(self, text_input):
        encoded_text = self.text_roberta(text_input)

        emo_vec = get_emo_vec(text_input)

        text = torch.cat((encoded_text, emo_vec), dim=1)  # check dim

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(text), self.dropout))

        # return the output
        return output


class MultitaskModel(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(self, params, device, model_type="roberta-base"):
        super(MultitaskModel, self).__init__()
        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = EmoLexiconModelBase(params, device, model_type)

        # set output layers
        self.sent_predictor = PredictionLayer(params, params.output_0_dim)
        self.emo_predictor = PredictionLayer(params, params.output_1_dim)

    def forward(
        self,
        text_input,
    ):
        # call forward on base model
        final_base_layer = self.base(text_input)

        sent_out = self.sent_predictor(final_base_layer)
        emo_out = self.emo_predictor(final_base_layer)

        return sent_out, emo_out