"""
PROGRAM:      root > environment.py
PROGRAMMER:   Jayra Gaile Ortiz
VERSION 1:    Sept. 21, 2022 
VERSION 2:    Jan. 03, 2023
PURPOSE:      Environments that are often used in the program
ALGORITHM:    Variable storing.

"""

from model.embedding import PretrainedEmbeddings
from model.layer import Bi_LSTM_CRF

word_embedding = "bert-base-uncased"
forward_embedding = "news-forward-fast"
backward_embedding = "news-backward-fast"

"""
embedding = PretrainedEmbeddings(
    word_embedding = word_embedding,
    forward_embedding=forward_embedding,
    backward_embedding= backward_embedding
).forward()"""

tagger = Bi_LSTM_CRF.load("checkpoints/best-model.pt")



roles = ["Doctor", "Researcher"]