import torch
import torch.nn as nn

from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    TFRobertaForSequenceClassification,
    pipeline
)


class EmoRoBERTa(nn.Module):
    def __init__(self, params, device):
        super(EmoRoBERTa).__init__()
        self.params = params
        self.device = device

        self.tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
        self.model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

        self.emotion = pipeline('sentiment-analysis',
                                model='arpanghoshal/EmoRoBERTa')

    def forward(self, text_inputs):
        all_inputs = []
        for text_input in text_inputs:
            all_inputs.append(self.emotion(text_input))

        return all_inputs


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