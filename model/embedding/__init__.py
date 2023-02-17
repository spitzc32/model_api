"""
PROGRAM:      init.py
PROGRAMMER:   Jayra Gaile Orti z
VERSION 1:    Aug. 29, 2022 
VERSION 2:    Nov. 30, 2022
PURPOSE:      Implementation of Embedding layers (FLAIR and RoBERTa)
ALGORITHM:    Initiating the context and tramsformer based embeddings and concatenating their outputs for the NER model  

"""


import torch.nn as nn
from flair.embeddings import (
    TransformerWordEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    StackedEmbeddings,
    OneHotEmbeddings
)
from flair.data import Sentence


class PretrainedEmbeddings():
    """
    This is the implmentation of the PretrainedEmbeddings we will use to embed our own
    corpus for the purpose of generating a Tensor(the pre_word_embeds) that we will pass
    to the model

    Plan:
    * Word-level Embeddings: We will utilize BERT Based transformer word embeddings 
    in order to achieve more functionality
    * Context-level Embeddings: We will stick to Flair Embeddings first then go check if
    pooled flair is better than FlairEmbeddings
    """

    def __init__(self, 
        word_embedding: str, 
        forward_embedding: str,
        backward_embedding: str
        ) -> None:
        self.word_embedding = word_embedding,
        self.forward_embedding = forward_embedding
        self.backward_embedding = backward_embedding

    
    def forward(self):
        # Firstly, we need to call out all pretrained embeddings accessible in
        # Flair for our requirement
        flair_forward_embedding = FlairEmbeddings(self.forward_embedding)
        flair_backward_embedding = FlairEmbeddings(self.backward_embedding)

        bert_embedding = TransformerWordEmbeddings(model=self.word_embedding,
                                       fine_tune=True,
                                       use_context=True,)

        # Next Concatenate all embeddings above
        stacked_embeddings = StackedEmbeddings(
            embeddings=[
                flair_forward_embedding, 
                flair_backward_embedding, 
                bert_embedding,
            ])

        return stacked_embeddings
        
       
           
 

    
    



