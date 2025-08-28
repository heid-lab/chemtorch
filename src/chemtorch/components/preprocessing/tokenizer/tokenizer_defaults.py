# regex taken from https://github.com/rxn4chemistry/rxnfp/blob/master/rxnfp/tokenization.py
DEFAULT_MOLECULE_PATTERN = (
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
    r"\?|\$|\|"             # don't know, kept for reproducability
    r")"
)
DEFAULT_UNK_TOKEN = "[UNK]"
DEFAULT_PAD_TOKEN = "[PAD]"
REACTION_SEPARATOR_TOKEN = ">>"
MOLECULE_SEPARATOR_TOKEN = "."