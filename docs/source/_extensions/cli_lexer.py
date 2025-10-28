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

    tokens = {
        'root': [
            # Comments
            (r'#.*$', Comment),
            
            # URLs/links (same as bash)
            (r'https?://[^\s]+', String.Other),
            
            # Package managers and tools (same color as bash commands like 'cd')
            (r'\b(conda|pip|uv|git)\b', Name.Builtin),
            
            # Numbers (complete numbers like 3.10, not splitting on dots)
            (r'\d+\.\d+', Number.Float),
            (r'\d+', Number.Integer),
            
            # Operators (same as bash)
            (r'[&|;(){}]', Operator),
            
            # Punctuation (same as bash)
            (r'[,.]', Punctuation),
            
            # Everything else as text
            (r'\S+', Text),
            (r'\s+', Whitespace),
        ]
    }

class ChemTorchCLILexer(RegexLexer):
    """
    A lexer for ChemTorch CLI commands with Hydra configuration.
    """
    
    name = 'ChemTorch CLI'
    aliases = ['chemtorch']
    filenames = []
    mimetypes = []

    tokens = {
        'root': [
            # ChemTorch CLI script name (now we know this pattern works)
            (r'chemtorch_cli\.py|chemtorch', Name.Exception),
            
            # Comments
            (r'#.*$', Comment),
            
            # Hydra config overrides - fix the ++ pattern (must come before single +)
            (r'(\+\+\w+(?:\.\w+)*|\+\w+(?:\.\w+)*|\w+\.\w+|\w+)(?==)', Generic.Inserted),
            
            # URLs/links (same as bash)
            (r'https?://[^\s]+', String.Other),
            
            # Python command (same color as bash commands)
            (r'\bpython\b', Name.Builtin),
            
            # Assignment operator
            (r'=', Operator),
            
            # Operators (same as bash)
            (r'[&|;(){}]', Operator),
            
            # Punctuation (same as bash)
            (r'[,.]', Punctuation),
            
            # Hydra flags (remain uncolored - just text)
            (r'(-m|--multi-run|-cn|--config-name|--config-path|-cp)', Text),
            
            # Values after = sign (remain uncolored)
            (r'(?<==)\S+', Text),
            
            # Numbers
            (r'\d+\.\d+', Number.Float),
            (r'\d+', Number.Integer),
            
            # Everything else as text
            (r'\S+', Text),
            (r'\s+', Whitespace),
        ]
    }

class WandbLexer(RegexLexer):
    """
    A lexer for Weights & Biases (wandb) commands.
    """
    
    name = 'Wandb'
    aliases = ['wandb']
    filenames = []
    mimetypes = []

    tokens = {
        'root': [
            # wandb command (same color as bash commands)
            (r'\bwandb\b', Name.Exception),
            
            # Subcommands (like 'login', 'init', 'sync', etc.)
            (r'\b(login|init|sweep|sync|agent|offline|online|watch|artifact)\b', Generic.Inserted),
            
            # Flags (like '--entity', '--project', etc.)
            (r'--\w+', Generic.Inserted),
            
            # # Values after flags (remain uncolored)
            # (r'(?<=--\w+\s)\S+', Text),
            
            # URLs/links (same as bash)
            (r'https?://[^\s]+', String.Other),
            
            # Comments
            (r'#.*$', Comment),
            
            # Numbers
            (r'\d+\.\d+', Number.Float),
            (r'\d+', Number.Integer),
            
            # Operators (same as bash)
            (r'[&|;(){}]', Operator),
            
            # Punctuation (same as bash)
            (r'[,.]', Punctuation),
            
            # Everything else as text
            (r'\S+', Text),
            (r'\s+', Whitespace),
        ]
    }

class WandbToHydraScriptLexer(RegexLexer):
    """
    A lexer for the wandb_to_hydra.py script command.
    """
    
    name = 'WandbToHydra'
    aliases = ['wandbtohydra']
    filenames = []
    mimetypes = []

    tokens = {
        'root': [
            # Script name
            (r'wandb_to_hydra\.py', Name.Exception),
            
            # Comments
            (r'#.*$', Comment),
            
            # Flags (like '--run-id', '--output-path', etc.)
            (r'--\w+(?:-\w+)*', Generic.Inserted),
            
            # Values after flags (remain uncolored)
            (r'(?<=--\w+(?:-\w+)*\s)\S+', Text),
            
            # URLs/links (same as bash)
            (r'https?://[^\s]+', String.Other),
            
            # Python command (same color as bash commands)
            (r'\bpython\b', Name.Builtin),
            
            # Operators (same as bash)
            (r'[&|;(){}]', Operator),
            
            # Punctuation (same as bash)
            (r'[,.]', Punctuation),
            
            # Numbers
            (r'\d+\.\d+', Number.Float),
            (r'\d+', Number.Integer),
            
            # Everything else as text
            (r'\S+', Text),
            (r'\s+', Whitespace),
        ]
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