import collections
from typing import Dict, List

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

import torch

from chemtorch.components.preprocessing.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.components.representation.token.abstract_token_representation import AbstractTokenRepresentation


class TokenRepresentationBase(AbstractTokenRepresentation):
    def __init__(
        self,
        tokenizer: AbstractTokenizer,
        max_sentence_length: int,
    ):
        self.max_sentence_length = max_sentence_length
        self.tokenizer = tokenizer

        self.token_to_id: Dict[str, int] = self._load_vocab(self.tokenizer.vocab_path)
        self.id_to_token: Dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}

        if self.tokenizer.pad_token not in self.token_to_id:
            raise ValueError(
                f"Pad token '{self.tokenizer.pad_token}' not found in vocabulary file: {self.tokenizer.vocab_path}"
            )
        self.pad_token_id: int = self.token_to_id[self.tokenizer.pad_token]

        if self.tokenizer.unk_token not in self.token_to_id:
            raise ValueError(
                f"UNK token '{self.tokenizer.unk_token}' not found in vocabulary file: {self.tokenizer.vocab_path}. "
                f"Ensure your vocab contains the UNK token used/defined by your tokenizer."
            )
        self.unk_token_id: int = self.token_to_id[self.tokenizer.unk_token]

    def _load_vocab(self, vocab_file_path: str) -> Dict[str, int]:
        """Loads a vocabulary file into an ordered dictionary."""
        word2id = collections.OrderedDict()
        with open(vocab_file_path, "r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                token = line.strip()
                if token:
                    if token in word2id:
                        print(
                            f"Warning: Duplicate token '{token}' found in vocab. Using first occurrence."
                        )
                    else:
                        word2id[token] = index
        return word2id

    def construct(self, smiles: str) -> torch.Tensor:
        """
        Tokenizes a SMILES string, converts to IDs, pads/truncates, and returns a 2D tensor.
        Output shape: (1, max_sentence_length)
        """
        if not isinstance(smiles, str):
            raise TypeError(f"Input 'smiles' must be a string, got {type(smiles)}")

        tokens = self.tokenizer.tokenize(smiles)

        if smiles and not tokens:
            tokens = [self.tokenizer.unk_token]

        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

        # pad or truncate
        if len(token_ids) < self.max_sentence_length:
            padding = [self.pad_token_id] * (self.max_sentence_length - len(token_ids))
            token_ids.extend(padding)
        else:
            token_ids = token_ids[: self.max_sentence_length]

        return torch.tensor([token_ids], dtype=torch.float)

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.token_to_id)
    
    @property 
    @override
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.token_to_id)
