import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.error import HTTPError

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import flair.nn
from part import *
from flair.data import Dictionary, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path
from flair.training_utils import store_embeddings

from model.layer.bioes import get_spans_from_bio
from model.layer.lstm import LSTM
from model.layer.crf import CRF
from model.layer.viterbi import ViterbiDecoder, ViterbiLoss

log = logging.getLogger("flair")


class Bi_LSTM_CRF(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        rnn: Optional[torch.nn.RNN] = None,
        tag_format: str = "BIOES",
        hidden_size: int = 256,
        rnn_layers: int = 1,
        bidirectional: bool = True,
        use_crf: bool = True,
        ave_embeddings: bool = True,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        loss_weights: Dict[str, float] = None,
        init_from_state_dict: bool = False,
        allow_unk_predictions: bool = False,
    ):
        """
        BiLSTM Span CRF class for predicting labels for single tokens. Can be parameterized by several attributes.
        Span prediction is utilized if there are nested entities such as Address and Organization. Since the researchers
        observed that the token are have different length for a given dataset, we made the Span useful by incorporating it 
        only if the data needs it. 

        :param embeddings: Embeddings to use during training and prediction
        :param tag_dictionary: Dictionary containing all tags from corpus which can be predicted
        :param tag_type: type of tag which is going to be predicted in case a corpus has multiple annotations
        :param rnn: (Optional) Takes a torch.nn.Module as parameter by which you can pass a shared RNN between
            different tasks.
        :param hidden_size: Hidden size of RNN layer
        :param rnn_layers: number of RNN layers
        :param bidirectional: If True, RNN becomes bidirectional
        :param use_crf: If True, use a Conditional Random Field for prediction, else linear map to tag space.
        :param ave_embeddings: If True, add a linear layer on top of embeddings, if you want to imitate
            fine tune non-trainable embeddings.
        :param dropout: If > 0, then use dropout.
        :param word_dropout: If > 0, then use word dropout.
        :param locked_dropout: If > 0, then use locked dropout.
        :param loss_weights: Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0)
        :param init_from_state_dict: Indicator whether we are loading a model from state dict
            since we need to transform previous models' weights into CRF instance weights
        """
        super(Bi_LSTM_CRF, self).__init__()

        # ----- Create the internal tag dictionary -----
        self.tag_type = tag_type
        self.tag_format = tag_format.upper()
        if init_from_state_dict:
            self.label_dictionary = tag_dictionary
        else:
            # span-labels need special encoding (BIO or BIOES)
            if tag_dictionary.span_labels:
                # the big question is whether the label dictionary should contain an UNK or not
                # without UNK, we cannot evaluate on data that contains labels not seen in test
                # with UNK, the model learns less well if there are no UNK examples
                self.label_dictionary = Dictionary(add_unk=allow_unk_predictions)
                assert self.tag_format in ["BIOES", "BIO"]
                for label in tag_dictionary.get_items():
                    if label == "<unk>":
                        continue
                    self.label_dictionary.add_item("O")
                    if self.tag_format == "BIOES":
                        self.label_dictionary.add_item("S-" + label)
                        self.label_dictionary.add_item("B-" + label)
                        self.label_dictionary.add_item("E-" + label)
                        self.label_dictionary.add_item("I-" + label)
                    if self.tag_format == "BIO":
                        self.label_dictionary.add_item("B-" + label)
                        self.label_dictionary.add_item("I-" + label)
            else:
                self.label_dictionary = tag_dictionary

        # is this a span prediction problem?
        self.predict_spans = self._determine_if_span_prediction_problem(self.label_dictionary)

        self.tagset_size = len(self.label_dictionary)
        log.info(f"SequenceTagger predicts: {self.label_dictionary}")

        # ----- Embeddings -----
        # We set the first initial embeddings gathered from Flair 
        # Stacked and concatenated then ave. using Linear
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length

        # ----- Initial loss weights parameters -----
        # This is for reiteration process of training.
        # Initially we don't have any loss weights, but as we proceed to training, 
        # we get loss computations from the evaluation stage.
        self.weight_dict = loss_weights
        self.loss_weights = self._init_loss_weights(loss_weights) if loss_weights else None

        # ----- RNN specific parameters -----
        # These parameters are for setting up the self.RNN 
        self.hidden_size = hidden_size if not rnn else rnn.hidden_size
        self.rnn_layers = rnn_layers if not rnn else rnn.num_layers
        self.bidirectional = bidirectional if not rnn else rnn.bidirectional

        # ----- Conditional Random Field parameters -----
        self.use_crf = use_crf
        # Previously trained models have been trained without an explicit CRF, thus it is required to check
        # whether we are loading a model from state dict in order to skip or add START and STOP token
        if use_crf and not init_from_state_dict and not self.label_dictionary.start_stop_tags_are_set():
            self.label_dictionary.set_start_stop_tags()
            self.tagset_size += 2

        # ----- Dropout parameters -----
        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        # ----- Model layers -----
        # Initialize Embedding Linear Dim for the purpose of ave them
        self.ave_embeddings = ave_embeddings
        if self.ave_embeddings:
            self.embedding2nn = torch.nn.Linear(embedding_dim, embedding_dim)

        # ----- RNN layer -----
        # If shared RNN provided, else create one for model
        self.rnn: torch.nn.RNN = (
            rnn
            if rnn
            else LSTM(
                rnn_layers,
                hidden_size,
                bidirectional,
                rnn_input_dim=embedding_dim,
            )
        )

        num_directions = 2 if self.bidirectional else 1
        hidden_output_dim = self.rnn.hidden_size * num_directions

     
        # final linear map to tag space
        self.linear = torch.nn.Linear(hidden_output_dim, len(self.label_dictionary))


        # the loss function is Viterbi if using CRF, else regular Cross Entropy Loss
        self.loss_function = (
            ViterbiLoss(self.label_dictionary)
        )

        # if using CRF, we also require a CRF and a Viterbi decoder
        if use_crf:
            self.crf = CRF(self.label_dictionary, self.tagset_size, init_from_state_dict)
            self.viterbi_decoder = ViterbiDecoder(self.label_dictionary)

        self.to(flair.device)

    @property
    def label_type(self):
        return self.tag_type

    def _init_loss_weights(self, loss_weights: Dict[str, float]) -> torch.Tensor:
        """
        Intializes the loss weights based on given dictionary:
        :param loss_weights: dictionary - contains loss weights
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1.0 for _ in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]

        return torch.tensor(weight_list).to(flair.device)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> Tuple[torch.Tensor, int]:
        """
        Calculates the loss of the forward propagation of the model
        :param sentences: either a listof sentence or just a sentence
        """
        # if there are no sentences, there is no loss
        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        # forward pass to get scores
        scores, gold_labels = self.forward(sentences)  # type: ignore

        # calculate loss given scores and labels
        return self._calculate_loss(scores, gold_labels)

    def forward(self, sentences: Union[List[Sentence], Sentence]):
        """
        Forward propagation through network. Returns gold labels of batch in addition.
        :param sentences: Batch of current sentences
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        self.embeddings.embed(sentences)

        # make a zero-padded tensor for the whole sentence
        lengths, sentence_tensor = self._make_padded_tensor_for_batch(sentences)

        # sort tensor in decreasing order based on lengths of sentences in batch
        sorted_lengths, length_indices = lengths.sort(dim=0, descending=True)
        sentences = [sentences[i] for i in length_indices]
        sentence_tensor = sentence_tensor[length_indices]

        # ----- Forward Propagation -----
        # we get the dropout we initialize for th regularization
        # of our inputs
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # Average the embeddings using Linear Transform
        if self.ave_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        # This packs our Sentence tensor form, the process for weighting
        # our LSTM model
        sentence_tensor, output_lengths = self.rnn(sentence_tensor, sorted_lengths)

        # Regularize our computed sentence tensor form the LSTM model
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # linear map to tag space
        features = self.linear(sentence_tensor)

        # Depending on whether we are using CRF or a linear layer, scores is either:
        # -- A tensor of shape (batch size, sequence length, tagset size, tagset size) for CRF
        # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
        if self.use_crf:
            features = self.crf(features)
            scores = (features, sorted_lengths, self.crf.transitions)
        else:
            scores = self._get_scores_from_features(features, sorted_lengths)

        # get the gold labels
        gold_labels = self._get_gold_labels(sentences)

        return scores, gold_labels

    def _calculate_loss(self, scores, labels) -> Tuple[torch.Tensor, int]:

        if not any(labels):
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        labels = torch.tensor(
            [
                self.label_dictionary.get_idx_for_item(label[0])
                if len(label) > 0
                else self.label_dictionary.get_idx_for_item("O")
                for label in labels
            ],
            dtype=torch.long,
            device=flair.device,
        )

        return self.loss_function(scores, labels), len(labels)

    def _make_padded_tensor_for_batch(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        makes zero padded tensors in the shape of the max longest sentence and the embedding_length to match 
        the shape of the embedding in feeding to our LSTM model.
        :param sentences: Batch of current sentences
        """
        names = self.embeddings.get_names()
        tok_lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(tok_lengths)
        zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )
        all_embs = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = zero_tensor[: self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )
        return torch.tensor(tok_lengths, dtype=torch.long), sentence_tensor

    @staticmethod
    def _get_scores_from_features(features: torch.Tensor, lengths: torch.Tensor):
        """
        Trims current batch tensor in shape (batch size, sequence length, tagset size) in such a way that all
        pads are going to be removed.
        :param features: torch.tensor containing all features from forward propagation
        :param lengths: length from each sentence in batch in order to trim padding tokens
        """
        features_formatted = []
        for feat, lens in zip(features, lengths):
            features_formatted.append(feat[:lens])
        scores = torch.cat(features_formatted)

        return scores

    def _get_gold_labels(self, sentences: Union[List[Sentence], Sentence]):
        """
        Extracts gold labels from each sentence.
        :param sentences: List of sentences in batch
        """
        # spans need to be encoded as token-level predictions
        if self.predict_spans:
            all_sentence_labels = []
            for sentence in sentences:
                sentence_labels = ["O"] * len(sentence)
                for label in sentence.get_labels(self.label_type):
                    span: Span = label.data_point
                    if self.tag_format == "BIOES":
                        if len(span) == 1:
                            sentence_labels[span[0].idx - 1] = "S-" + label.value
                        else:
                            sentence_labels[span[0].idx - 1] = "B-" + label.value
                            sentence_labels[span[-1].idx - 1] = "E-" + label.value
                            for i in range(span[0].idx, span[-1].idx - 1):
                                sentence_labels[i] = "I-" + label.value
                    else:
                        sentence_labels[span[0].idx - 1] = "B-" + label.value
                        for i in range(span[0].idx, span[-1].idx):
                            sentence_labels[i] = "I-" + label.value
                all_sentence_labels.extend(sentence_labels)
            labels = [[label] for label in all_sentence_labels]

        # all others are regular labels for each token
        else:
            labels = [[token.get_label(self.label_type, "O").value] for sentence in sentences for token in sentence]

        return labels

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        force_token_predictions: bool = False,
    ):  # type: ignore
        """
        Predicts labels for current batch with CRF.
        :param sentences: List of sentences in batch
        :param mini_batch_size: batch size for test data
        :param return_probabilities_for_all_classes: Whether to return probabilites for all classes
        :param verbose: whether to use progress bar
        :param label_name: which label to predict
        :param return_loss: whether to return loss value
        :param embedding_storage_mode: determines where to store embeddings - can be "gpu", "cpu" or None.
        """
        if label_name is None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            # make sure its a list
            if not isinstance(sentences, list) and not isinstance(sentences, flair.data.Dataset):
                sentences = [sentences]

            # filter empty sentences
            sentences = [sentence for sentence in sentences if len(sentence) > 0]

            # reverse sort all sequences by their length
            reordered_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)

            if len(reordered_sentences) == 0:
                return sentences

            dataloader = DataLoader(
                dataset=FlairDatapointDataset(reordered_sentences),
                batch_size=mini_batch_size,
            )
            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader, desc="Batch inference")

            overall_loss = torch.zeros(1, device=flair.device)
            batch_no = 0
            label_count = 0
            for batch in dataloader:

                batch_no += 1

                # stop if all sentences are empty
                if not batch:
                    continue

                # get features from forward propagation
                features, gold_labels = self.forward(batch)

                # remove previously predicted labels of this type
                for sentence in batch:
                    sentence.remove_labels(label_name)

                # if return_loss, get loss value
                if return_loss:
                    loss = self._calculate_loss(features, gold_labels)
                    overall_loss += loss[0]
                    label_count += loss[1]

                # Sort batch in same way as forward propagation
                lengths = torch.LongTensor([len(sentence) for sentence in batch])
                _, sort_indices = lengths.sort(dim=0, descending=True)
                batch = [batch[i] for i in sort_indices]

                # make predictions
                if self.use_crf:
                    predictions, all_tags = self.viterbi_decoder.decode(
                        features, return_probabilities_for_all_classes, batch
                    )
                else:
                    predictions, all_tags = self._standard_inference(
                        features, batch, return_probabilities_for_all_classes
                    )

                # add predictions to Sentence
                for sentence, sentence_predictions in zip(batch, predictions):

                    # BIOES-labels need to be converted to spans
                    if self.predict_spans and not force_token_predictions:
                        sentence_tags = [label[0] for label in sentence_predictions]
                        sentence_scores = [label[1] for label in sentence_predictions]
                        predicted_spans = get_spans_from_bio(sentence_tags, sentence_scores)
                        for predicted_span in predicted_spans:
                            span: Span = sentence[predicted_span[0][0] : predicted_span[0][-1] + 1]
                            span.add_label(label_name, value=predicted_span[2], score=predicted_span[1])

                    # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
                    else:
                        for token, label in zip(sentence.tokens, sentence_predictions):
                            if label[0] in ["O", "_"]:
                                continue
                            token.add_label(typename=label_name, value=label[0], score=label[1])

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                store_embeddings(sentences, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count

    def _standard_inference(self, features: torch.Tensor, batch: List[Sentence], probabilities_for_all_classes: bool):
        """
        Softmax over emission scores from forward propagation.
        :param features: sentence tensor from forward propagation
        :param batch: list of sentence
        :param probabilities_for_all_classes: whether to return score for each tag in tag dictionary
        """
        softmax_batch = F.softmax(features, dim=1).cpu()
        scores_batch, prediction_batch = torch.max(softmax_batch, dim=1)
        predictions = []
        all_tags = []

        for sentence in batch:
            scores = scores_batch[: len(sentence)]
            predictions_for_sentence = prediction_batch[: len(sentence)]
            predictions.append(
                [
                    (self.label_dictionary.get_item_for_index(prediction), score.item())
                    for token, score, prediction in zip(sentence, scores, predictions_for_sentence)
                ]
            )
            scores_batch = scores_batch[len(sentence) :]
            prediction_batch = prediction_batch[len(sentence) :]

        if probabilities_for_all_classes:
            lengths = [len(sentence) for sentence in batch]
            all_tags = self._all_scores_for_token(batch, softmax_batch, lengths)

        return predictions, all_tags

    def _all_scores_for_token(self, sentences: List[Sentence], scores: torch.Tensor, lengths: List[int]):
        """
        Returns all scores for each tag in tag dictionary.
        :param scores: Scores for current sentence.
        """
        scores = scores.numpy()
        tokens = [token for sentence in sentences for token in sentence]
        prob_all_tags = [
            [
                Label(token, self.label_dictionary.get_item_for_index(score_id), score)
                for score_id, score in enumerate(score_dist)
            ]
            for score_dist, token in zip(scores, tokens)
        ]

        prob_tags_per_sentence = []
        previous = 0
        for length in lengths:
            prob_tags_per_sentence.append(prob_all_tags[previous : previous + length])
            previous = length
        return prob_tags_per_sentence

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "tag_dictionary": self.label_dictionary,
            "tag_format": self.tag_format,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "ave_embeddings": self.ave_embeddings,
            "weight_dict": self.weight_dict,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        if state["use_crf"]:
            if "transitions" in state["state_dict"]:
                state["state_dict"]["crf.transitions"] = state["state_dict"]["transitions"]
                del state["state_dict"]["transitions"]

        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            tag_dictionary=state.get("tag_dictionary"),
            tag_format=state.get("tag_format", "BIOES"),
            tag_type=state.get("tag_type"),
            use_crf=state.get("use_crf"),
            rnn_layers=state.get("rnn_layers"),
            hidden_size=state.get("hidden_size"),
            dropout=state.get("use_dropout", 0.0),
            word_dropout=state.get("use_word_dropout", 0.0),
            locked_dropout=state.get("use_locked_dropout", 0.0),
            ave_embeddings=state.get("ave_embeddings", True),
            loss_weights=state.get("weight_dict"),
            init_from_state_dict=True,
            **kwargs,
        )

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens.")
        return filtered_sentences

    def _determine_if_span_prediction_problem(self, dictionary: Dictionary) -> bool:
        for item in dictionary.get_items():
            if item.startswith("B-") or item.startswith("S-") or item.startswith("I-"):
                return True
        return False

    def _print_predictions(self, batch, gold_label_type):

        lines = []
        if self.predict_spans:
            for datapoint in batch:
                # all labels default to "O"
                for token in datapoint:
                    token.set_label("gold_bio", "O")
                    token.set_label("predicted_bio", "O")

                # set gold token-level
                for gold_label in datapoint.get_labels(gold_label_type):
                    gold_span: Span = gold_label.data_point
                    prefix = "B-"
                    for token in gold_span:
                        token.set_label("gold_bio", prefix + gold_label.value)
                        prefix = "I-"

                # set predicted token-level
                for predicted_label in datapoint.get_labels("predicted"):
                    predicted_span: Span = predicted_label.data_point
                    prefix = "B-"
                    for token in predicted_span:
                        token.set_label("predicted_bio", prefix + predicted_label.value)
                        prefix = "I-"

                # now print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label('gold_bio').value} "
                        f"{token.get_label('predicted_bio').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")

        else:
            for datapoint in batch:
                # print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label(gold_label_type).value} "
                        f"{token.get_label('predicted').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")
        return lines

