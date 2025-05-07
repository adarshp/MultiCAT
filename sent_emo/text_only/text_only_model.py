# baseline model for sentiment and emotion classification

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
)


class RobertaFromTFBase(nn.Module):
    """
    For Roberta models that have a TF base
    """

    def __init__(self, device, model_type):
        super(RobertaBase, self).__init__()

        self.device = device

        self.model = RobertaModel.from_pretrained(model_type, from_tf=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_type)

    def forward(self, text_inputs):
        # tokenize and complete forward pass with roberta over the batch of inputs
        # holders for tokenized data
        batch_ids = []
        batch_masks = []
        # tokenize each item
        for item in text_inputs:
            if type(item) == list:
                item = " ".join(item)
            text = self.tokenizer(
                item,
                add_special_tokens=True,
                truncation=True,
                max_length=64,  # 256,
                padding="max_length",
            )
            # add to holder
            batch_ids.append(text["input_ids"])
            batch_masks.append(text["attention_mask"])

        # convert to tensor for use with model
        batch_ids = torch.tensor(batch_ids).to(self.device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks).to(self.device, dtype=torch.long)

        # feed through the model
        roberta_out = self.model(input_ids=batch_ids, attention_mask=batch_masks)

        # return either pooled output or last hidden state for cls token
        # to get pooled output, use roberta_out['pooler_output']
        # to get cls last hidden, use roberta_out['last_hidden_state][:, 0, :]
        return roberta_out["pooler_output"]


class RobertaBase(nn.Module):
    def __init__(self, device, model_type, config):
        super(RobertaBase, self).__init__()

        self.device = device

        self.robertaconfig = RobertaConfig(
            hidden_dropout_prob=config.dropout,
            vocab_size=50265,
            max_position_embeddings=514,
            type_vocab_size=1,
        )
        try:
            self.model = RobertaModel.from_pretrained(
                model_type, config=self.robertaconfig
            )
        except OSError:
            print("Successfully got TF model EmoRoBERTa")
            self.model = RobertaModel.from_pretrained(model_type, from_tf=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_type)

    def forward(self, text_inputs):
        # tokenize and complete forward pass with roberta over the batch of inputs
        # holders for tokenized data
        batch_ids = []
        batch_masks = []
        # tokenize each item
        for item in text_inputs:
            if type(item) == list:
                item = " ".join(item)
            text = self.tokenizer(
                item,
                add_special_tokens=True,
                truncation=True,
                max_length=64,  # 256,
                padding="max_length",
            )
            # add to holder
            batch_ids.append(text["input_ids"])
            batch_masks.append(text["attention_mask"])

        # convert to tensor for use with model
        batch_ids = torch.tensor(batch_ids).to(self.device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks).to(self.device, dtype=torch.long)

        # feed through the model
        roberta_out = self.model(input_ids=batch_ids, attention_mask=batch_masks)

        # return either pooled output or last hidden state for cls token
        # to get pooled output, use roberta_out['pooler_output']
        # to get cls last hidden, use roberta_out['last_hidden_state][:, 0, :]
        return roberta_out["pooler_output"]


class TextModelBase(nn.Module):
    """
    An encoder to take a sequence of inputs and
    produce a sequence of intermediate representations
    """

    def __init__(self, params, device, model_type="roberta-base"):
        super(TextModelBase, self).__init__()
        # set roberta base
        self.text_roberta = RobertaBase(
            device=device, model_type=model_type, config=params
        )

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize fully connected layers
        self.fc1 = nn.Linear(params.text_dim, params.fc_hidden_dim)

    def forward(self, text_input):
        encoded_text = self.text_roberta(text_input)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(encoded_text), self.dropout))

        # return the output
        return output


class PredictionLayer(nn.Module):
    """
    A final layer for predictions
    """

    def __init__(self, params, out_dim):
        super(PredictionLayer, self).__init__()
        self.input_dim = params.fc_hidden_dim
        self.inter_fc_prediction_dim = params.final_hidden_dim
        self.dropout = params.dropout

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        # self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.fc1 = nn.Linear(self.input_dim, self.inter_fc_prediction_dim)
        self.fc2 = nn.Linear(self.inter_fc_prediction_dim, self.output_dim)

    def forward(self, combined_inputs):
        out = torch.relu(F.dropout(self.fc1(combined_inputs), self.dropout))
        out = torch.relu(self.fc2(out))

        return out


class MultitaskModel(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(self, params, device, model_type="roberta-base"):
        super(MultitaskModel, self).__init__()
        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = TextModelBase(params, device, model_type)

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


class SingleTaskModel(nn.Module):
    """
    A single-task model; can be used to just get one type of prediction
    Or for separate models for each task
    """

    def __init__(self, params, device, model_type="roberta-base"):
        super(SingleTaskModel, self).__init__()
        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = TextModelBase(params, device, model_type)

        # set output layers
        if params.model.lower() == "emotion" or params.model.lower() == "emo":
            self.predictor = PredictionLayer(params, params.output_1_dim)
        else:
            self.predictor = PredictionLayer(params, params.output_0_dim)

    def forward(
        self,
        text_input,
    ):
        # call forward on base model
        final_base_layer = self.base(text_input)

        out = self.predictor(final_base_layer)

        return out