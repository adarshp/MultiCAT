# baseline model for sentiment and emotion classification

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    RobertaTokenizer,
    RobertaModel,
)


class RobertaBase(nn.Module):
    def __init__(self, device):
        super(RobertaBase, self).__init__()

        self.device = device

        # self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
                max_length=256,  # 256,
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
        # return roberta_out['pooler_output']
        return roberta_out["last_hidden_state"][:, 0, :]


class MultimodalModelBase(nn.Module):
    """
    An encoder to take a sequence of inputs and
    produce a sequence of intermediate representations
    todo: try replacing text rnn with BERT model
    """

    def __init__(self, params, device):
        super(MultimodalModelBase, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # set device
        self.device = device

        # get number of output dims
        self.out_dims = params.output_dim

        self.text_roberta = RobertaBase(device=device)

        # todo: implement me as a test
        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # # set size of input dim
        # self.fc_input_dim = (
        #     params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim
        # )
        # set size of hidden
        self.fc_hidden = 100
        # set acoustic fc layer 1
        ## todo: change after testing
        # self.acoustic_fc_1 = nn.Linear(params.audio_dim, self.fc_hidden)
        ## self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, self.fc_hidden)

        ## set acoustic fc layer 2
        # self.acoustic_fc_2 = nn.Linear(self.fc_hidden, params.audio_dim)

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # set size of input into fc
        # self.fc_input_dim = params.text_dim + params.audio_dim
        self.fc_input_dim = params.text_dim + params.acoustic_gru_hidden_dim

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        # self.fc_batch_norm = nn.BatchNorm1d(params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        length_input=None,
        acoustic_len_input=None,
    ):
        # # flatten_parameters() decreases memory usage
        # self.text_rnn.flatten_parameters()

        # if type(text_input) == torch.tensor:
        #     packed = nn.utils.rnn.pack_padded_sequence(
        #         text_input, length_input, batch_first=True, enforce_sorted=False
        #     )
        #     # feed embeddings through GRU
        #     packed_output, (hidden, cell) = self.text_rnn(packed)
        #     encoded_text = F.dropout(hidden[-1], 0.3)
        # else:
        # call to roberta model
        encoded_text = self.text_roberta(text_input)

        if acoustic_len_input is not None:
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                acoustic_input,
                acoustic_len_input.clamp(max=500),  # max=1500
                batch_first=True,
                enforce_sorted=False,
            )
            (
                packed_acoustic_output,
                (acoustic_hidden, acoustic_cell),
            ) = self.acoustic_rnn(packed_acoustic.to(self.device))
            encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)
        # else:
        #   if len(acoustic_input.shape) > 2:
        #       # get average of dim 1, as this is the un-averaged acoustic info
        #       encoded_acoustic = torch.mean(acoustic_input, dim=1)
        #   else:
        #       encoded_acoustic = acoustic_input

        # encoded_acoustic = torch.tanh(
        #    F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout)
        # )
        # encoded_acoustic = torch.tanh(
        #    F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
        # )

        # combine modalities as required by architecture
        inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

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

    def __init__(self, params, device):
        super(MultitaskModel, self).__init__()
        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = MultimodalModelBase(params, device)

        # set output layers
        self.sent_predictor = PredictionLayer(params, params.output_0_dim)
        self.emo_predictor = PredictionLayer(params, params.output_1_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        length_input=None,
        acoustic_len_input=None,
    ):
        # call forward on base model
        final_base_layer = self.base(
            acoustic_input,
            text_input,
            length_input=length_input,
            acoustic_len_input=acoustic_len_input,
        )

        sent_out = self.sent_predictor(final_base_layer)
        emo_out = self.emo_predictor(final_base_layer)

        return sent_out, emo_out
