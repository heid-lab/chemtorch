"""
Custom Pygments lexers for ChemTorch documentation.
"""

import re
from pygments.lexer import RegexLexer
from pygments.token import (
    Text, Comment, Keyword, Name, String, Number, Operator, Punctuation,
    Generic, Whitespace
)


class InstallationLexer(RegexLexer):
    """
    A lexer for installation commands highlighting package managers.
    """

    name = 'Installation'
    aliases = ['install']
    filenames = []
    mimetypes = []


class ChemTorchCLILexer(RegexLexer):
    """
    A lexer for ChemTorch CLI commands with Hydra configuration.
    """

    name = 'ChemTorch CLI'
    aliases = ['chemtorch']
    filenames = []
    mimetypes = []


class WandbLexer(RegexLexer):
    """
    A lexer for Weights & Biases (wandb) commands.
    """

    name = 'Wandb'
    aliases = ['wandb']
    filenames = []
    mimetypes = []


class WandbToHydraScriptLexer(RegexLexer):
    """
    A lexer for the wandb_to_hydra.py script command.
    """

    name = 'WandbToHydra'
    aliases = ['wandbtohydra']
    filenames = []
    mimetypes = []


# Shared/common token patterns used by multiple lexers. Keeping these
# centralized reduces redundancy and ensures consistent ordering.
COMMON_ROOT = [
    # Comments
    (r'#.*$', Comment),

    # URLs/links
    (r'https?://[^\s]+', String.Other),

    # Numbers (floats then ints)
    (r'\d+\.\d+', Number.Float),
    (r'\d+', Number.Integer),

    # Operators and punctuation
    (r'[&|;(){}]', Operator),
    (r'[,.]', Punctuation),

    # Default fallthrough
    (r'\S+', Text),
    (r'\s+', Whitespace),
]


# Specific token sets for each lexer state. Specific patterns should come
# before the common patterns so they get matched first.
InstallationLexer.tokens = {
    'root': [
        # Package managers and tools (highlight as builtin commands)
        (r'\b(conda|pip|uv|git)\b', Name.Builtin),
    ] + COMMON_ROOT
}

# We only want to highlight 'chemtorch' when it appears as a standalone
# command/token (start of line or preceded by whitespace), not when it is
# embedded inside a file path. Use lookarounds to require start/whitespace
# before and whitespace/end after. We use two fixed-width lookbehinds to
# satisfy Python's regex engine.
_CHEMTORCH_CMD = r'(?:(?<=\s)|(?<=^))(?:chemtorch_cli\.py|chemtorch)(?=(?:\s|$))'

ChemTorchCLILexer.tokens = {
    'root': [
        # ChemTorch CLI script name (only when standalone)
        (_CHEMTORCH_CMD, Name.Exception),

        # Hydra config overrides - ++ must come before single +
        (r'(\+\+\w+(?:\.\w+)*|\+\w+(?:\.\w+)*|\w+\.\w+|\w+)(?==)', Generic.Inserted),

        # Python command
        (r'\bpython\b', Name.Builtin),

        # Assignment operator and other operators
        (r'=', Operator),
        (r'[&|;(){}]', Operator),
        (r'[,.]', Punctuation),

        # Hydra flags (keep as plain text)
        (r'(-m|--multi-run|-cn|--config-name|--config-path|-cp)', Text),

        # Values after '=' sign (keep as plain text)
        (r'(?<==)\S+', Text),
    ] + COMMON_ROOT
}

WandbLexer.tokens = {
    'root': [
        # wandb command only when it appears as a separate token
        (r'(?:(?<=\s)|(?<=^))wandb(?=(?:\s|$))', Name.Exception),

        # Subcommands (like 'login', 'init', 'sync', etc.)
        (r'\b(login|init|sweep|sync|agent|offline|online|watch|artifact)\b', Generic.Inserted),

        # Flags (like '--entity', '--project', etc.)
        (r'--\w+', Generic.Inserted),
    ] + COMMON_ROOT
}

WandbToHydraScriptLexer.tokens = {
    'root': [
        # Script name
        (r'(?:(?<=\s)|(?<=^))wandb_to_hydra\.py(?=(?:\s|$))', Name.Exception),

        # Flags (like '--run-id', '--output-path', etc.)
        (r'--\w+(?:-\w+)*', Generic.Inserted),

        # Values after flags (remain uncolored)
        (r'(?<=--\w+(?:-\w+)*\s)\S+', Text),

        # Python command
        (r'\bpython\b', Name.Builtin),
    ] + COMMON_ROOT
}


def setup(app):
    """Setup function for the Sphinx extension"""
    # Register lexers
    app.add_lexer('install', InstallationLexer)
    app.add_lexer('chemtorch', ChemTorchCLILexer)
    app.add_lexer('wandb', WandbLexer)
    app.add_lexer('wandbtohydra', WandbToHydraScriptLexer)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }