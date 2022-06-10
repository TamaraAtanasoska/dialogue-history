from types import SimpleNamespace

import torch
import torch.nn as nn

from lxmert.src.lxrt.entry import LXRTEncoder
from models.Decider import Decider
from models.Guesser import Guesser

"""
Putting all the models together
"""
class LXMERTEnsembleGuesserOnly(nn.Module):
    """docstring for Ensemble."""
    def __init__(self, **kwargs):
        super(LXMERTEnsembleGuesserOnly, self).__init__()
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
        self.ensemble_args = kwargs

        # TODO: use get_attr to get different versions of the same model. For example QGen

        lxrt_encoder_args = SimpleNamespace(**self.ensemble_args["encoder"]["LXRTEncoder"])
        self.lxrt_encoder = LXRTEncoder(
            lxrt_encoder_args,
            max_seq_length=200
        )
        if not lxrt_encoder_args.from_scratch:
            print("Loading LXMERT pretrained model...")
            self.lxrt_encoder.load(lxrt_encoder_args.model_path)
        else:
            print("Initializing LXMERT model from scratch...")

        self.guesser = Guesser(**self.ensemble_args['guesser'])

        self.decider = Decider(**self.ensemble_args['decider'])

        self.dropout = nn.Dropout(p=0.5)

        self.scale_to = nn.Sequential(
            nn.Linear(768, self.ensemble_args['encoder']['scale_to']),
            nn.Tanh()
        )

    def forward(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'history_raw' :
            'fasterrcnn_features' :
            'fasterrcnn_boxes' :
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
        spatials = kwargs['spatials']
        objects = kwargs['objects']

        lxrt_res = self.lxrt_encoder(kwargs["history_raw"], (kwargs["fasterrcnn_features"], kwargs["fasterrcnn_boxes"]))
        encoder_hidden = self.scale_to(torch.unsqueeze(lxrt_res, 1))

        decider_out = self.decider(encoder_hidden=encoder_hidden)

        guesser_out = self.guesser(encoder_hidden=encoder_hidden, spatials=spatials, objects=objects, regress=False)

        return decider_out, guesser_out
