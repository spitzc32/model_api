from typing import Dict, List, Optional
from flair.data import _PartOfSentence, DataPoint, Label

class Token(_PartOfSentence):
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.

    :param text: Single text(Token) from the sequence
    :param head_id: the location of the text (For Document)
    :param whitespace_after: if token has whitespace
    :param start_position: what character number in document does this token start?
    :param sentence: If token belongs to sentence, indicate here which var it belongs to
    """

    def __init__(
        self,
        text: str,
        head_id: int = None,
        whitespace_after: int = 1,
        start_position: int = 0,
        sentence=None,
    ):
        super().__init__(sentence=sentence)

        self.form: str = text
        self._internal_index: Optional[int] = None
        self.head_id: Optional[int] = head_id
        self.whitespace_after: int = whitespace_after

        self.start_pos = start_position
        self.end_pos = start_position + len(text)

        self._embeddings: Dict = {}
        self.tags_proba_dist: Dict[str, List[Label]] = {}

    @property
    def idx(self) -> int:
        if isinstance(self._internal_index, int):
            return self._internal_index
        else:
            raise ValueError

    @property
    def text(self):
        return self.form

    @property
    def unlabeled_identifier(self) -> str:
        return f'Token[{self.idx-1}]: "{self.text}"'

    def add_tags_proba_dist(self, tag_type: str, tags: List[Label]):
        self.tags_proba_dist[tag_type] = tags

    def get_tags_proba_dist(self, tag_type: str) -> List[Label]:
        if tag_type in self.tags_proba_dist:
            return self.tags_proba_dist[tag_type]
        return []

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    @property
    def start_position(self) -> int:
        return self.start_pos

    @property
    def end_position(self) -> int:
        return self.end_pos

    @property
    def embedding(self):
        return self.get_embedding()

    def __repr__(self):
        return self.__str__()

    def add_label(self, typename: str, value: str, score: float = 1.0):
        """
        The Token is a special _PartOfSentence in that it may be initialized without a Sentence.
        Therefore, labels get added only to the Sentence if it exists
        """
        if self.sentence:
            super().add_label(typename=typename, value=value, score=score)
        else:
            DataPoint.add_label(self, typename=typename, value=value, score=score)

    def set_label(self, typename: str, value: str, score: float = 1.0):
        """
        The Token is a special _PartOfSentence in that it may be initialized without a Sentence.
        Therefore, labels get set only to the Sentence if it exists
        """
        if self.sentence:
            super().set_label(typename=typename, value=value, score=score)
        else:
            DataPoint.set_label(self, typename=typename, value=value, score=score)


class Span(_PartOfSentence):
    """
    This class represents one textual span consisting of Tokens. It may be used for the instance that the 
    tokens form in a nested nature, meaning the tokens combined together forms a long phrase.

    :param tokens: List of tokens in the span
    """

    def __init__(self, tokens: List[Token]):
        super().__init__(tokens[0].sentence)
        self.tokens = tokens
        super()._init_labels()

    @property
    def start_position(self) -> int:
        return self.tokens[0].start_position

    @property
    def end_position(self) -> int:
        return self.tokens[-1].end_position

    @property
    def text(self) -> str:
        return " ".join([t.text for t in self.tokens])

    @property
    def unlabeled_identifier(self) -> str:
        return f'Span[{self.tokens[0].idx -1}:{self.tokens[-1].idx}]: "{self.text}"'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def embedding(self):
        pass