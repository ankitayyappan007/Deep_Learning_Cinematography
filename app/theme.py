COLORS = {
    "background": "#0B0D14",
    "sidebar": "#111522",
    "panel": "#171B29",
    "panel_alt": "#1C2233",
    "border": "#30364A",
    "text": "#F2F2FA",
    "muted": "#A7ADC3",
    "accent": "#A99AFF",
    "cyan": "#65D9D0",
    "warning": "#FFD18A",
    "warning_bg": "#30251F",
    "success": "#86E0B0",
    "success_bg": "#172B29",
    "selected": "#272340",
    "track": "#242A3A",
}


def configure_style(root):
    from tkinter import ttk

    root.option_add("*Font", ("Segoe UI", 10))
    root.option_add("*Foreground", COLORS["text"])
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(
        "TFrame",
        background=COLORS["background"],
    )
    style.configure(
        "TLabel",
        background=COLORS["background"],
        foreground=COLORS["text"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "TButton",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        padding=(12, 9),
        borderwidth=0,
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "TButton",
        background=[
            ("active", COLORS["panel_alt"]),
            ("pressed", COLORS["selected"]),
        ],
        foreground=[("active", COLORS["cyan"])],
    )

    style.configure(
        "Nav.TButton",
        background=COLORS["sidebar"],
        foreground=COLORS["muted"],
        anchor="w",
        padding=(16, 12),
        borderwidth=0,
        font=("Segoe UI", 10),
    )
    style.map(
        "Nav.TButton",
        background=[
            ("selected", COLORS["selected"]),
            ("active", COLORS["panel"]),
        ],
        foreground=[
            ("selected", COLORS["accent"]),
            ("active", COLORS["text"]),
        ],
    )

    style.configure(
        "Treeview",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        fieldbackground=COLORS["panel"],
        rowheight=34,
        borderwidth=0,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["sidebar"],
        foreground=COLORS["cyan"],
        padding=(10, 11),
        font=("Segoe UI", 9, "bold"),
        borderwidth=0,
    )
    style.map(
        "Treeview",
        background=[("selected", COLORS["selected"])],
        foreground=[("selected", COLORS["text"])],
    )

    style.configure(
        "Vertical.TScrollbar",
        background=COLORS["panel_alt"],
        troughcolor=COLORS["background"],
        bordercolor=COLORS["background"],
        arrowcolor=COLORS["muted"],
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=COLORS["panel_alt"],
        troughcolor=COLORS["background"],
        bordercolor=COLORS["background"],
        arrowcolor=COLORS["muted"],
    )
