import torch.nn as nn

from models.EncoderDeVries import EncoderDeVries
from models.Guesser import Guesser

"""
Putting all the models together
"""
class EnsembleDeVries(nn.Module):
    """docstring for Ensemble."""
    def __init__(self, **kwargs):
        super(EnsembleDeVries, self).__init__()
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

        self.encoder = EncoderDeVries(**self.ensemble_args['encoder'])

        self.guesser = Guesser(**self.ensemble_args['guesser'])

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
        history, history_len = kwargs['history'], kwargs['history_len']
        spatials = kwargs['spatials']
        objects = kwargs['objects']
        encoder_hidden = self.encoder(history=history, history_len=history_len)
        guesser_out = self.guesser(encoder_hidden=encoder_hidden, spatials=spatials, objects=objects, regress=False)
        return guesser_out
