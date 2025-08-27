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
    r"\%[0-9]{2}|"          # Percent with 2 digits for ring closure with more than 9 rings open at once (extremely rare)
    r")"
)
DEFAULT_UNK_TOKEN = "[UNK]"
DEFAULT_PAD_TOKEN = "[PAD]"
REACTION_SEPARATOR_TOKEN = ">>"
MOLECULE_SEPARATOR_TOKEN = "."