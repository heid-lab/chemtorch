from chemtorch.components.representation.token.tokenizer.molecule_tokenizer.regex_tokenizer import RegexTokenizer


# adapted from https://github.com/rxn4chemistry/rxnfp/blob/master/rxnfp/tokenization.py
RXNFP_REGEX_PATTERN = (
    r"("
    r"\[[^\]]+]|"           # Bracketed expressions: [CH3], [OH], [N+]
    r"Br?|Cl?|"             # Bromine/Chlorine (with optional second letter)
    r"N|O|S|P|F|I|"         # Single uppercase atoms
    r"b|c|n|o|s|p|"         # Single lowercase atoms
    r"\(|\)|\.|"            # Structural symbols: |, (, ), .
    r"-|=|#|:|"             # Bond symbols: single (-), double (=), triple (#), aromatic (:)
    r"\\|\/|\+|@|"          # Special symbols: configuration around double bonds (\ and /), sterics (@ or @@)
    r">>?|"                 # Reaction arrows: > or >>
    r"~|\*|"                # Wildcards: any bond (~), any atom (*)
    r"[0-9]"                # Single digits for ring closure
    r"\%[0-9]{2}|"          # Ring closure with more than 9 rings open at once (extremely rare)
    r"\%\([0-9]{3}\)|"      # Ring closure with more than 99 rings open at once (even rarer, kept for reproducability)
    r"\?|\$|\|"             # don't know (maybe SMARTS syntax?), kept for reproducability
    r")"
)

class SmilesSymbolTokenizer(RegexTokenizer):
    """
    Tokenizes a SMILES string into its elementary symbols based on the RXNFP regex pattern.
    
    Paper: https://www.nature.com/articles/s42256-020-00284-w
    """

    def __init__(self, vocab_path: str, unk_token: str, pad_token: str):
        super().__init__(regex_pattern=RXNFP_REGEX_PATTERN, vocab_path=vocab_path, unk_token=unk_token, pad_token=pad_token)