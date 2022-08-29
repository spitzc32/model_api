from functools import lru_cache
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def enumerate_spans(n):
    for i in range(n):
        for j in range(i, n):
            yield (i, j)

@lru_cache  # type: ignore
def get_all_spans(n: int) -> torch.Tensor:
    return torch.tensor(list(enumerate_spans(n)), dtype=torch.long)


class SpanClassifier(nn.Module):
    num_additional_labels = 1

    def __init__(self, encoder, scorer: "SpanScorer"):
        super().__init__()
        self.encoder = encoder
        self.scorer = scorer

    def forward(
        self, *input_ids: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        hs, lengths = self.encoder(*input_ids)
        spans = list(map(get_all_spans, lengths))
        scores = self.scorer(hs, spans)
        return spans, scores

    @torch.no_grad()
    def decode(
        self,
        spans: Sequence[torch.Tensor],
        scores: Sequence[torch.Tensor],
    ) -> List[List[Tuple[int, int, int]]]:
        spans_flatten = torch.cat(spans)
        scores_flatten = torch.cat(scores)
        assert len(spans_flatten) == len(scores_flatten)
        labels_flatten = scores_flatten.argmax(dim=1).cpu()
        mask = labels_flatten < self.scorer.num_labels - 1
        mentions = torch.hstack((spans_flatten[mask], labels_flatten[mask, None]))

        output = []
        offset = 0
        sizes = [m.sum() for m in torch.split(mask, [len(idxs) for idxs in spans])]
        for size in sizes:
            output.append([tuple(m) for m in mentions[offset : offset + size].tolist()])
            offset += size
        return output  # type: ignore

    def compute_metrics(
        self,
        spans: Sequence[torch.Tensor],
        scores: Sequence[torch.Tensor],
        true_mentions: Sequence[Sequence[Tuple[int, int, int]]],
        decode=True,
    ) -> Dict[str, Any]:
        assert len(spans) == len(scores) == len(true_mentions)
        num_labels = self.scorer.num_labels
        true_labels = []
        for spans_i, scores_i, true_mentions_i in zip(spans, scores, true_mentions):
            assert len(spans_i) == len(scores_i)
            span2idx = {tuple(s): idx for idx, s in enumerate(spans_i.tolist())}
            labels_i = torch.full((len(spans_i),), fill_value=num_labels - 1)
            for (start, end, label) in true_mentions_i:
                idx = span2idx.get((start, end))
                if idx is not None:
                    labels_i[idx] = label
            true_labels.append(labels_i)

        scores_flatten = torch.cat(scores)
        true_labels_flatten = torch.cat(true_labels).to(scores_flatten.device)
        assert len(scores_flatten) == len(true_labels_flatten)
        loss = F.cross_entropy(scores_flatten, true_labels_flatten)
        accuracy = categorical_accuracy(scores_flatten, true_labels_flatten)
        result = {"loss": loss, "accuracy": accuracy}

        if decode:
            pred_mentions = self.decode(spans, scores)
            tp, fn, fp = 0, 0, 0
            for pred_mentions_i, true_mentions_i in zip(pred_mentions, true_mentions):
                pred, gold = set(pred_mentions_i), set(true_mentions_i)
                tp += len(gold & pred)
                fn += len(gold - pred)
                fp += len(pred - gold)
            result["precision"] = (tp, tp + fp)
            result["recall"] = (tp, tp + fn)
            result["mentions"] = pred_mentions

        return result


@torch.no_grad()
def categorical_accuracy(
    y: torch.Tensor, t: torch.Tensor, ignore_index: Optional[int] = None
) -> Tuple[int, int]:
    pred = y.argmax(dim=1)
    if ignore_index is not None:
        mask = t == ignore_index
        ignore_cnt = mask.sum()
        pred.masked_fill_(mask, ignore_index)
        count = ((pred == t).sum() - ignore_cnt).item()
        total = (t.numel() - ignore_cnt).item()
    else:
        count = (pred == t).sum().item()
        total = t.numel()
    return count, total


class SpanScorer(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

    def forward(
        self, xs: torch.Tensor, spans: Sequence[torch.Tensor]
    ):
        raise NotImplementedError


class BaselineSpanScorer(SpanScorer):
    def __init__(
        self,
        input_size: int,
        num_labels: int,
        mlp_units: Union[int, Sequence[int]] = 150,
        mlp_dropout: float = 0.0,
        feature="concat",
    ):
        super().__init__(num_labels)
        input_size *= 2 if feature == "concat" else 1
        self.mlp = MLP(input_size, num_labels, mlp_units, F.relu, mlp_dropout)
        self.feature = feature

    def forward(
        self, xs: torch.Tensor, spans: Sequence[torch.Tensor]
    ):
        max_length = xs.size(1)
        xs_flatten = xs.reshape(-1, xs.size(-1))
        spans_flatten = torch.cat([idxs + max_length * i for i, idxs in enumerate(spans)])
        features = self._compute_feature(xs_flatten, spans_flatten)
        scores = self.mlp(features)
        return torch.split(scores, [len(idxs) for idxs in spans])

    def _compute_feature(self, xs, spans):
        if self.feature == "concat":
            return xs[spans.ravel()].view(len(spans), -1)
        elif self.feature == "minus":
            begins, ends = spans.T
            return xs[ends] - xs[begins]
        else:
            raise NotImplementedError


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int],
        units: Optional[Union[int, Sequence[int]]] = None,
        activate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        units = [units] if isinstance(units, int) else units
        if not units and out_features is None:
            raise ValueError("'out_features' or 'units' must be specified")
        layers = []
        for u in units or []:
            layers.append(MLP.Layer(in_features, u, activate, dropout, bias))
            in_features = u
        if out_features is not None:
            layers.append(MLP.Layer(in_features, out_features, None, 0.0, bias))
        super().__init__(*layers)

    class Layer(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            activate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            dropout: float = 0.0,
            bias: bool = True,
        ):
            super().__init__()
            if activate is not None and not callable(activate):
                raise TypeError("activate must be callable: type={}".format(type(activate)))
            self.linear = nn.Linear(in_features, out_features, bias)
            self.activate = activate
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.linear(x)
            if self.activate is not None:
                h = self.activate(h)
            return self.dropout(h)

        def extra_repr(self) -> str:
            return "{}, activate={}, dropout={}".format(
                self.linear.extra_repr(), self.activate, self.dropout.p
            )

        def __repr__(self):
            return "{}.{}({})".format(MLP.__name__, self._get_name(), self.extra_repr())

