"""
PROGRAM:      crf.py
PROGRAMMER:   Jayra Gaile Ortiz
VERSION 1:    Aug. 29, 2022 
PURPOSE:      Implementation of CRF model
ALGORITHM:    In accordance to sgrvinod,
              Forward propagation


"""



import torch

import flair

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


class CRF(torch.nn.Module):
    """
    Conditional Random Field Implementation according to sgrvinod and modified to not 
    only look at the current word, but also on the previously seen annotation.
    """

    def __init__(self, tag_dictionary, tagset_size: int, init_from_state_dict: bool):
        """
        :param tag_dictionary: tag dictionary in order to find ID for start and stop tags
        :param tagset_size: number of tag from tag dictionary
        :param init_from_state_dict: whether we load pretrained model from state dict
        """
        super(CRF, self).__init__()

        self.tagset_size = tagset_size
        # Transitions are used in the following way: transitions[to, from].
        self.transitions = torch.nn.Parameter(torch.randn(tagset_size, tagset_size))
        # If we are not using a pretrained model and train a fresh one, we need to set transitions from any tag
        # to START-tag and from STOP-tag to any other tag to -10000.
        if not init_from_state_dict:
            self.transitions.detach()[tag_dictionary.get_idx_for_item(START_TAG), :] = -10000

            self.transitions.detach()[:, tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000
        self.to(flair.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of Conditional Random Field.
        :param features: output from LSTM Layer in shape (batch size, seq len, hidden size)
        :return: CRF scores (emission scores for each token + transitions prob from previous state) in
        shape (batch_size, seq len, tagset size, tagset size)
        """
        batch_size, seq_len = features.size()[:2]

        emission_scores = features
        emission_scores = emission_scores.unsqueeze(-1).expand(batch_size, seq_len, self.tagset_size, self.tagset_size)

        crf_scores = emission_scores + self.transitions.unsqueeze(0).unsqueeze(0)
        return crf_scores