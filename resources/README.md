The `base_vocab.txt` vocabulary was sourced from the [rxnfp BERT model](https://github.com/rxn4chemistry/rxnfp/blob/master/rxnfp/models/transformers/bert_class_1k_tpl/vocab.txt).
The original vocabulary was extended by the following tokens which occur in `uspto_1k`, `rdb7` and `rgd1` (see our [GitHub](https://github.com/heid-lab/reaction_database/tree/main)) but not in the original vocabulary:
```
[Sn+6]
[n]
[o]
[N-2]
[O+2]
```

The `substruc_vocab.txt` was created by extending `base_vocab.txt` with the tokens obtained from parsing all reactions in the `cycloadd`, `rdb7`, `rgd1`, `sn2`, `e2`, and `uspto-1` datasets using the `SubstructureTokenizer` (see `src/chemtorch/components/preprocessing/tokenizer/molecule_tokenizer/substructure_tokenizer.py`).