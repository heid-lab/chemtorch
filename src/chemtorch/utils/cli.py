
CHEMTORCH_LOGO_LINES: list[str] = [
    " ██████╗██╗  ██╗███████╗███╗   ███╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗",
    "██╔════╝██║  ██║██╔════╝████╗ ████║╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║",
    "██║     ███████║█████╗  ██╔████╔██║   ██║   ██║   ██║██████╔╝██║     ███████║",
    "██║     ██╔══██║██╔══╝  ██║╚██╔╝██║   ██║   ██║   ██║██╔══██╗██║     ██╔══██║",
    "╚██████╗██║  ██║███████╗██║ ╚═╝ ██║   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║",
    " ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝",
]

# Custom color gradient from orange to dark purple (using RGB values)
GRADIENT_COLORS: list[str] = [
    "\033[38;2;255;128;0m",   # #ff8000 - bright orange
    "\033[38;2;223;64;32m",   # #df4020 - orange-red
    "\033[38;2;191;0;64m",    # #bf0040 - red-magenta
    "\033[38;2;160;0;96m",    # #a00060 - purple-magenta
    "\033[38;2;89;0;89m",     # #590059 - dark purple
    "\033[38;2;89;0;89m",     # #590059 - dark purple
]


def cli_chemtorch_logo() -> None:
    print()
    for i, line in enumerate(CHEMTORCH_LOGO_LINES):
        color = GRADIENT_COLORS[i % len(GRADIENT_COLORS)]
        print(f"{color}{line}\033[0m")
    print()