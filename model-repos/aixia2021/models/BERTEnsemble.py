import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

from models.Guesser import Guesser


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        input_ids = tokenizer.encode(tokens_a)
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # In Roberta the padding is equal to 1
        padding = [1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def convert_sents_to_features_roberta(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        sent = sent.strip().lower()
        if not sent:
            sent = [""]

        encodings = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_seq_length)

        input_ids = encodings["input_ids"]
        input_mask = encodings["attention_mask"]

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=None))
    return features


"""
Putting all the models together
"""


class BERTEnsemble(nn.Module):
    """docstring for Ensemble."""

    def __init__(self, max_seq_length=200, from_scratch=False, **kwargs):
        super(BERTEnsemble, self).__init__()
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'encoder' : Arguments for the encoder module
            'qgen' : Arguments for the qgen module
            'guesser' : Arguments for the guesser module
            'regressor' : Arguments for the regressor module
            'decider' : Arguments for the decider module

        """
        self.max_seq_length = max_seq_length

        self.ensemble_args = kwargs

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder = RobertaModel.from_pretrained("roberta-base", output_attentions=True)

        if from_scratch:
            print("Training from scratch...")
            self.encoder = RobertaModel(self.encoder.config)

        self.guesser = Guesser(**self.ensemble_args['guesser'])

        self.scale_to = nn.Sequential(
            nn.Linear(768, self.ensemble_args['encoder']['scale_to']),
            nn.Tanh()
        )

    def forward(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'history' : The dialogue history. Shape :[Bx max_src_length]
            'history_len' : The length of the dialogue history. Shape [Bx1]
            'src_q' : The input word sequence for the QGen
            'tgt_len' : Length of the target question
            'visual_features' : The avg pool layer from ResNet 152
            'spatials' : Spatial features for the guesser. Shape [Bx20x8]
            'objects' : List of objects for guesser. Shape [Bx20]
            'mask_select' : Bool. Based on the decider target, either QGen or Guesser is used

        Returns
        -------
        ensemble_out : dict
            'decider_out' : predicted decision
            'guesser_out' : log probabilities of the objects
            'qgen_out' : predicted next question

        """
        history_raw = kwargs["history_raw"]
        spatials = kwargs['spatials']
        objects = kwargs['objects']

        features = convert_sents_to_features_roberta(history_raw, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.float).cuda()
        # segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).cuda()

        # for cls:
        # encoder_hidden = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=None)[0][:, 0, :]
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=None)
        encoder_hidden = encoder_output[1]
        self.encoder_attentions = encoder_output[2]
        encoder_hidden = self.scale_to(encoder_hidden)
        guesser_out = self.guesser(encoder_hidden=encoder_hidden, spatials=spatials, objects=objects, regress=False)
        return guesser_out
