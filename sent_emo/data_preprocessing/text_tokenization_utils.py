import torch
from torch import nn
from transformers import (
    BertTokenizer,
    BertModel,
    DistilBertModel,
    DistilBertTokenizer,
    RobertaTokenizer,
    RobertaModel,
)


class BertEmb:
    def __init__(self, use_roberta=False):
        if not use_roberta:
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base", do_lower_case=True
            )

        # Load pre-trained model (weights)
        if not use_roberta:
            self.model = BertModel.from_pretrained(
                "bert-base-uncased",
                output_hidden_states=True,
            )
        else:
            self.model = RobertaModel.from_pretrained(
                "roberta-base", output_hidden_states=True
            )
        self.model.eval()

    def tokenize(self, sent):
        """
        Tokenize a sentence with bert tokenizer; convert to indices
        :param sent: A sentence to tokenize
        :return: tokenized sentence and corresponding vector of 1s
        """
        sent = "[CLS] " + sent + " [SEP]"
        tokens = self.tokenizer.tokenize(sent)

        # Map the token strings to their vocabulary indeces.
        idx_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        idx_tokens = torch.tensor(idx_tokens)
        idx_tokens = idx_tokens.unsqueeze(0)
        # can use this creatively during fine-tuning
        ids = [1] * len(tokens)

        return idx_tokens, ids

    def get_embeddings(self, utt_tensor, id_tensor, longest_utt=None):
        """
        For a tensor of ALL utts from ALL datasets
        Get embeddings from penultimate layer of bert
        :param utt_tensor: a tensor of tokenized utterances
        :param id_tensor: a tensor of ids for utterances (0 or 1)
        todo: ids_tensor is currently all 1s bc of the nature of our data
        :return:
        """
        # get holder for embeddings
        embeddings = []

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.
        with torch.no_grad():
            outputs = self.model(utt_tensor, id_tensor)

            # for BERT (but NOT distilbert):
            # with `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_state = outputs[2]
            # Concatenate the tensors for all layers. We use `stack` here to
            # create a new dimension in the tensor.
            token_embeddings = torch.stack(hidden_state, dim=0)
            # Remove dimension 1, the "batches".
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            # Swap dimensions 0 and 1.
            token_embeddings = token_embeddings.permute(1, 0, 2)
            # # get word embeddings
            for emb_layer_mat in token_embeddings:
                # select penultimate hidden layer as embedding
                token_emb = emb_layer_mat[-2]
                # add this embedding to word counts and add to idx2emb
                embeddings.append(token_emb)

        embeddings = torch.stack(embeddings)
        embeddings = torch.squeeze(embeddings, dim=0)
        embeddings = torch.squeeze(embeddings, dim=1)

        # pad to make the correct length
        # based on the difference between max_len and embeddings dim 0
        if longest_utt:
            # add zeros to the end of the first dimension
            padding = (0, 0, 0, longest_utt - embeddings.shape[0])
            embeddings = nn.functional.pad(embeddings, padding, "constant", 0)

        return embeddings


class DistilBertEmb:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", do_lower_case=True
        )

        # Load pre-trained model (weights)
        self.model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased",
            output_hidden_states=True,
        )
        self.model.eval()

    def tokenize(self, sent):
        """
        Tokenize a sentence with bert tokenizer; convert to indices
        :param sent: A sentence to tokenize
        :return: tokenized sentence and corresponding vector of 1s
        """
        sent = "[CLS] " + sent + " [SEP]"
        tokens = self.tokenizer.tokenize(sent)

        # Map the token strings to their vocabulary indeces.
        idx_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        idx_tokens = torch.tensor(idx_tokens)
        idx_tokens = idx_tokens.unsqueeze(0)
        # can use this creatively during fine-tuning
        ids = [1] * len(tokens)

        return idx_tokens, ids

    def get_embeddings(self, utt_tensor, id_tensor, longest_utt=None):
        """
        For a tensor of ALL utts from ALL datasets
        Get embeddings from penultimate layer of bert
        :param utt_tensor: a tensor of tokenized utterances
        :param id_tensor: a tensor of ids for utterances (0 or 1)
        todo: ids_tensor is currently all 1s bc of the nature of our data
        :return:
        """
        # get holder for embeddings
        embeddings = []

        # Run the text through distilBERT, and collect all hidden states produced
        # from all 12 layers.
        with torch.no_grad():
            outputs = self.model(utt_tensor, id_tensor)

            # for BERT (but NOT distilbert):
            # with `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            # output 0 is the output of the final layer of distilbert
            hidden_state = outputs[0]

            # Swap dimensions 0 and 1.
            token_embeddings = hidden_state.permute(1, 0, 2)

            # append to embeddings
            embeddings.append(token_embeddings)

        embeddings = torch.stack(embeddings)
        embeddings = torch.squeeze(embeddings, dim=0)
        embeddings = torch.squeeze(embeddings, dim=1)

        # pad to make the correct length
        # based on the difference between max_len and embeddings dim 0
        if longest_utt:
            # add zeros to the end of the first dimension
            padding = (0, 0, 0, longest_utt - embeddings.shape[0])
            embeddings = nn.functional.pad(embeddings, padding, "constant", 0)

        return embeddings


def get_distilbert_tokenizer():
    # instantiate distilbert emb object
    distilbert_emb_mkr = DistilBertEmb()

    # return tokenizer
    return distilbert_emb_mkr.tokenizer


def get_bert_tokenizer():
    # instantiate bert emb object
    bert_emb_mkr = BertEmb()

    # return
    return bert_emb_mkr.tokenizer


def get_roberta_tokenizer():
    # instantiate roberta emb object
    roberta_emb_mkr = BertEmb(use_roberta=True)

    # return
    return roberta_emb_mkr.tokenizer


def get_longest_utt(data, tokenizer, use_bert=False, return_text=False):
    """
    Get the length of the longest utterance in the dataset
    :param data: a pandas df containing all utterances
    :param tokenizer: a tokenizer
    :param use_bert: whether to use bert/distilbert tokenizer
    :param return_text: whether the data created will have text (vs embeddings)
    :return: length of the longest utterance
    """
    if "utterance" in data.columns.to_list():
        all_utts = data["utterance"].tolist()
    else:
        all_utts = data["utt"].tolist()

    if return_text:
        return 1
    elif use_bert:
        # tokenize and count all items in dataset
        item_lens = [
            len(tokenizer.tokenize("[CLS] " + str(utt) + " [SEP]")) for utt in all_utts
        ]
    else:
        # tokenize, clean up, and count all items in dataset
        item_lens = [len(tokenizer(clean_up_word(str(utt)))) for utt in all_utts]

    # return longest
    return max(item_lens)


def clean_up_word(word):
    word = word.replace("\x92", "'")
    word = word.replace("\x91", " ")
    word = word.replace("\x97", "-")
    word = word.replace("\x93", " ")
    word = word.replace("[", " ")
    word = word.replace("]", " ")
    word = word.replace("-", " - ")
    word = word.replace("%", " % ")
    word = word.replace("@", " @ ")
    word = word.replace("$", " $ ")
    word = word.replace("...", " ... ")
    word = word.replace("/", " / ")
    word = word.replace(".", " . ")
    word = word.replace(",", " , ")
    word = word.replace("!", " ! ")
    word = word.replace("?", " ? ")
    if word.strip() == "":
        word = "<UNK>"
    return word
