import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import flair

class LSTM(torch.nn.Module):
    """
    Simple LSTM Implementation that returns the features used for (1)CRF and (2)Span Classifier

    """
    def __init__(self, rnn_layers: int, hidden_size: int, bidirectional: bool, rnn_input_dim: int,):
        """
        :param rnn_layers: number of rnn layers to be used, default 1
        :param hidden_size: hidden size of the LSTM layer
        :param bidirectional: whether we use biderectional lstm or not, default True
        :param rnn_input_dim: the shape of our max sentence token and embeddings 
        """
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_input_dim = rnn_input_dim
        self.num_layers = rnn_layers
        self.dropout = 0.0 if rnn_layers == 1 else 0.5
        self.bidirectional = bidirectional
        self.batch_first = True
        self.lstm = torch.nn.LSTM(
            self.rnn_input_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
        )

        self.to(flair.device)
    
    def forward(self, sentence_tensor: torch.Tensor, sorted_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of LSTM Model by packing the tensors.
        :param features: output from RNN / Linear layer in shape (batch size, seq len, hidden size)
        :return: CRF scores (emission scores for each token + transitions prob from previous state) in
        shape (batch_size, seq len, tagset size, tagset size)
        """
        packed = pack_padded_sequence(sentence_tensor, sorted_lengths, batch_first=True, enforce_sorted=False)
        rnn_output, hidden = self.lstm(packed)
        sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)

        return sentence_tensor, output_lengths
