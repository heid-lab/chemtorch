import collections
from typing import Dict, List

import torch

from chemtorch.components.preprocessing.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.components.preprocessing.tokenizer.tokenizer_defaults import DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN
from chemtorch.components.representation.token.abstract_token_representation import AbstractTokenRepresentation


class TokenRepresentationBase(AbstractTokenRepresentation):
    def __init__(
        self,
        vocab_path: str,
        tokenizer: AbstractTokenizer,
        max_sentence_length: int,
        pad_token: str = DEFAULT_PAD_TOKEN,
        unk_token: str = DEFAULT_UNK_TOKEN,
        *args,
        **kwargs,
    ):
        self.max_sentence_length = max_sentence_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.tokenizer = tokenizer

        self._word2id: Dict[str, int] = self._load_vocab(vocab_path)
        self._id2word: Dict[int, str] = {
            idx: token for token, idx in self._word2id.items()
        }

        if self.pad_token not in self._word2id:
            raise ValueError(
                f"Pad token '{self.pad_token}' not found in vocabulary file: {vocab_path}"
            )
        self.pad_token_id: int = self._word2id[self.pad_token]

        if self.unk_token not in self._word2id:
            raise ValueError(
                f"UNK token '{self.unk_token}' not found in vocabulary file: {vocab_path}. "
                f"Ensure your vocab contains the UNK token used/defined by your tokenizer."
            )
        self.unk_token_id: int = self._word2id[self.unk_token]

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

        tokens: List[str] = self.tokenizer.tokenize(smiles)

        token_ids: List[int] = [
            self._word2id.get(token, self.unk_token_id) for token in tokens
        ]

        # pad or truncate
        if len(token_ids) < self.max_sentence_length:
            padding = [self.pad_token_id] * (self.max_sentence_length - len(token_ids))
            token_ids.extend(padding)
        else:
            token_ids = token_ids[: self.max_sentence_length]

        return torch.tensor([token_ids], dtype=torch.float)

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self._word2id)
    
    @property 
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self._word2id)

    @property
    def word2id(self) -> Dict[str, int]:
        """Returns the word to ID mapping."""
        return self._word2id
    
    @property
    def id2word(self) -> Dict[int, str]:
        """Returns the ID to word mapping."""
        return self._id2word
