from chemtorch.components.representation.token.tokenizer.abstract_tokenizer import AbstractTokenizer

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

class MoleculeTokenizerBase(AbstractTokenizer):
    def __init__(self, vocab_path: str, unk_token: str, pad_token: str):
        self._vocab_path = vocab_path
        self._unk_token = unk_token
        self._pad_token = pad_token

    @property
    @override
    def vocab_path(self) -> str:
        """Path to the vocabulary file."""
        return self._vocab_path

    @property
    @override
    def pad_token(self) -> str:
        """Token to use for padding."""
        return self._pad_token

    @property
    @override
    def unk_token(self) -> str:
        """Token to use for unknown tokens."""
        return self._unk_token