from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from data_loader import load_all, format_percent
from theme import COLORS, configure_style


class CineScopeApp(tk.Tk):
    NAV_ITEMS = (
        ("Dashboard", "dashboard"),
        ("Shot-Scale Explorer", "shot_scale"),
        ("Camera-Movement Explorer", "movement"),
        ("Data Status", "status"),
    )

    def __init__(self):
        super().__init__()
        self.title("CineScope AI | Cinematography Intelligence")
        self.geometry("1380x860")
        self.minsize(1020, 680)
        self.configure(bg=COLORS["background"])
        self.data = load_all()
        self._nav_buttons = {}
        configure_style(self)
        self._build_shell()
        self.show_dashboard()

    def _build_shell(self):
        self.shell = tk.Frame(self, bg=COLORS["background"])
        self.shell.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self.shell, bg=COLORS["sidebar"], width=245)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        brand.pack(fill="x", padx=20, pady=(26, 22))
        logo = tk.Canvas(
            brand, width=42, height=42, bg=COLORS["sidebar"],
            highlightthickness=0
        )
        logo.pack(anchor="w", pady=(0, 12))
        logo.create_oval(3, 3, 39, 39, outline=COLORS["accent"], width=2)
        logo.create_oval(10, 10, 32, 32, outline=COLORS["cyan"], width=1)
        logo.create_line(21, 5, 21, 37, fill=COLORS["accent"], width=1)
        logo.create_line(5, 21, 37, 21, fill=COLORS["accent"], width=1)
        logo.create_oval(17, 17, 25, 25, fill=COLORS["accent"], outline="")
        tk.Label(
            brand, text="CineScope", bg=COLORS["sidebar"],
            fg=COLORS["text"], font=("Segoe UI", 21, "bold")
        ).pack(anchor="w")
        tk.Label(
            brand, text="AI  /  CINEMATOGRAPHY", bg=COLORS["sidebar"],
            fg=COLORS["cyan"], font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", pady=(3, 0))
        tk.Frame(self.sidebar, bg=COLORS["border"], height=1).pack(
            fill="x", padx=18, pady=(0, 18)
        )
        tk.Label(
            self.sidebar, text="WORKSPACE", bg=COLORS["sidebar"],
            fg=COLORS["muted"], font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=22, pady=(0, 8))

        for label, page in self.NAV_ITEMS:
            button = tk.Button(
                self.sidebar, text=label, anchor="w",
                bg=COLORS["sidebar"], fg=COLORS["muted"],
                activebackground=COLORS["selected"],
                activeforeground=COLORS["text"],
                relief="flat", bd=0, padx=16, pady=12,
                font=("Segoe UI", 10), cursor="hand2",
                command=lambda selected=page: self._navigate(selected)
            )
            button.pack(fill="x", padx=12, pady=3)
            self._nav_buttons[page] = button

        spacer = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        spacer.pack(fill="both", expand=True)
        tk.Frame(self.sidebar, bg=COLORS["border"], height=1).pack(
            fill="x", padx=18, pady=(0, 12)
        )
        tk.Label(
            self.sidebar, text="EVALUATION WORKSPACE",
            bg=COLORS["sidebar"], fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=21)
        tk.Label(
            self.sidebar, text="LOCAL ARTIFACTS  |  NO TRAINING",
            bg=COLORS["sidebar"], fg=COLORS["cyan"],
            font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", padx=21, pady=(5, 22))

        self.content = tk.Frame(self.shell, bg=COLORS["background"])
        self.content.pack(side="left", fill="both", expand=True)

    def _navigate(self, page):
        for name, button in self._nav_buttons.items():
            selected = name == page
            button.configure(
                bg=COLORS["selected"] if selected else COLORS["sidebar"],
                fg=COLORS["accent"] if selected else COLORS["muted"],
                font=("Segoe UI", 10, "bold" if selected else "normal"),
            )
        if page == "dashboard":
            self.show_dashboard()
        elif page in ("shot_scale", "movement"):
            self.show_explorer(page)
        elif page == "status":
            self.show_status()

    def _clear(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def _header(self, title, subtitle, eyebrow="CINEMATOGRAPHY INTELLIGENCE"):
        header = tk.Frame(self.content, bg=COLORS["background"])
        header.pack(fill="x", padx=30, pady=(24, 17))
        tk.Label(
            header, text=eyebrow, bg=COLORS["background"],
            fg=COLORS["accent"], font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", pady=(0, 5))
        tk.Label(
            header, text=title, bg=COLORS["background"],
            fg=COLORS["text"], font=("Segoe UI", 23, "bold")
        ).pack(anchor="w")
        tk.Label(
            header, text=subtitle, bg=COLORS["background"],
            fg=COLORS["muted"], font=("Segoe UI", 10),
            wraplength=1000, justify="left"
        ).pack(anchor="w", pady=(5, 0))
        tk.Frame(self.content, bg=COLORS["border"], height=1).pack(
            fill="x", padx=30, pady=(0, 15)
        )

    def _scroll_area(self):
        outer = tk.Frame(self.content, bg=COLORS["background"])
        outer.pack(fill="both", expand=True, padx=30, pady=(0, 22))
        canvas = tk.Canvas(
            outer, bg=COLORS["background"], highlightthickness=0,
            borderwidth=0
        )
        scrollbar = ttk.Scrollbar(
            outer, orient="vertical", command=canvas.yview
        )
        inner = tk.Frame(canvas, bg=COLORS["background"])
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def update_scroll(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def fit_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        def wheel(event):
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120) or
                                    (-1 if event.delta > 0 else 1), "units")
            return "break"

        inner.bind("<Configure>", update_scroll)
        canvas.bind("<Configure>", fit_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<MouseWheel>", wheel)
        inner.bind("<MouseWheel>", wheel)
        return inner

    def _section(self, parent, title, subtitle=None):
        frame = tk.Frame(parent, bg=COLORS["background"])
        frame.pack(fill="x", pady=(17, 7))
        tk.Frame(frame, bg=COLORS["accent"], width=3, height=23).pack(
            side="left", padx=(0, 10)
        )
        labels = tk.Frame(frame, bg=COLORS["background"])
        labels.pack(side="left", fill="x", expand=True)
        tk.Label(
            labels, text=title, bg=COLORS["background"],
            fg=COLORS["text"], font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        if subtitle:
            tk.Label(
                labels, text=subtitle, bg=COLORS["background"],
                fg=COLORS["muted"], font=("Segoe UI", 9),
                wraplength=1000, justify="left"
            ).pack(anchor="w", pady=(3, 0))

    def _panel(self, parent, padding=16):
        return tk.Frame(
            parent, bg=COLORS["panel"],
            highlightbackground=COLORS["border"], highlightthickness=1,
            padx=padding, pady=padding
        )

    def _cards(self, parent, cards, columns=4):
        row = tk.Frame(parent, bg=COLORS["background"])
        row.pack(fill="x", pady=7)
        for index, (label, value, detail) in enumerate(cards):
            column = index % columns
            line = index // columns
            card = self._panel(row, padding=14)
            card.grid(
                row=line, column=column, sticky="nsew",
                padx=(0 if column == 0 else 6, 6 if column < columns - 1 else 0),
                pady=(0, 10)
            )
            row.columnconfigure(column, weight=1, uniform="metric")
            tk.Label(
                card, text=label.upper(), bg=COLORS["panel"],
                fg=COLORS["muted"], font=("Segoe UI", 8, "bold")
            ).pack(anchor="w")
            tk.Label(
                card, text=str(value), bg=COLORS["panel"],
                fg=COLORS["accent"], font=("Segoe UI", 21, "bold")
            ).pack(anchor="w", pady=(8, 4))
            tk.Label(
                card, text=detail, bg=COLORS["panel"],
                fg=COLORS["muted"], font=("Segoe UI", 9),
                wraplength=210, justify="left"
            ).pack(anchor="w")

    def _tree(self, parent, columns, rows, headings=None, height=8):
        frame = self._panel(parent, padding=7)
        frame.pack(fill="x", pady=7)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=height
        )
        for index, column in enumerate(columns):
            heading = headings[index] if headings else column.replace("_", " ").title()
            tree.heading(column, text=heading)
            width = 180 if column in ("task", "experiment", "source") else 115
            tree.column(column, width=width, minwidth=85, anchor="center",
                        stretch=True)
        for row in rows:
            tree.insert("", "end", values=row)
        ybar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        if len(rows) <= height:
            ybar.grid_remove()
        return tree

    def _warnings(self, parent, warnings):
        if not warnings:
            return
        self._section(parent, "Data validation notices",
                      "Review these messages before interpreting incomplete results.")
        for warning in warnings:
            item = tk.Frame(
                parent, bg=COLORS["warning_bg"],
                highlightbackground=COLORS["warning"], highlightthickness=1
            )
            item.pack(fill="x", pady=4)
            tk.Label(
                item, text="!", bg=COLORS["warning_bg"],
                fg=COLORS["warning"], font=("Segoe UI", 12, "bold"),
                padx=12, pady=8
            ).pack(side="left", anchor="n")
            tk.Label(
                item, text=str(warning), bg=COLORS["warning_bg"],
                fg=COLORS["text"], font=("Segoe UI", 9),
                wraplength=950, justify="left", padx=5, pady=10
            ).pack(side="left", fill="x", expand=True)

    def _comparison_rows(self):
        rows = []
        for task, collection in (
            ("Shot scale", self.data["shot_scale"]["experiments"]),
            ("Camera movement", self.data["movement"]["experiments"]),
        ):
            for name, experiment in collection.items():
                metrics = experiment.get("metrics", {})
                rows.append((
                    task, experiment.get("label", name),
                    format_percent(metrics.get("test_accuracy")),
                    format_percent(metrics.get("macro_f1")),
                    format_percent(metrics.get("weighted_f1")),
                    metrics.get("test_samples", metrics.get("test_frames", "?")),
                ))
        return rows

    def _comparison_chart(self, parent, title, entries, metric_key,
                          metric_label, note):
        panel = self._panel(parent, padding=15)
        panel.pack(fill="x", pady=7)
        tk.Label(
            panel, text=title, bg=COLORS["panel"],
            fg=COLORS["text"], font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")
        tk.Label(
            panel, text=note, bg=COLORS["panel"],
            fg=COLORS["muted"], font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(3, 12))

        for label, metrics, color in entries:
            line = tk.Frame(panel, bg=COLORS["panel"])
            line.pack(fill="x", pady=5)
            tk.Label(
                line, text=label, bg=COLORS["panel"],
                fg=COLORS["muted"], font=("Segoe UI", 9),
                width=27, anchor="w"
            ).pack(side="left")
            track = tk.Canvas(
                line, height=18, bg=COLORS["track"],
                highlightthickness=0, bd=0
            )
            track.pack(side="left", fill="x", expand=True, padx=10)
            raw = metrics.get(metric_key)
            try:
                number = float(raw)
                valid = 0 <= number <= 1
            except (TypeError, ValueError):
                number, valid = 0.0, False

            def draw_bar(event, canvas=track, value=number, fill=color,
                         okay=valid):
                canvas.delete("all")
                width = max(0, event.width)
                if okay and width:
                    canvas.create_rectangle(
                        0, 3, width * value, 15, fill=fill, outline=""
                    )
            track.bind("<Configure>", draw_bar)
            tk.Label(
                line, text=format_percent(raw), bg=COLORS["panel"],
                fg=COLORS["text"] if valid else COLORS["muted"],
                font=("Segoe UI", 10, "bold"), width=9, anchor="e"
            ).pack(side="right")

    def show_dashboard(self):
        self._navigate_highlight("dashboard")
        self._clear()
        self._header(
            "Cinematography Intelligence",
            "Saved experiment results  |  Local evaluation workspace  |  No training or inference",
            "CINESCOPE AI  /  OVERVIEW"
        )
        body = self._scroll_area()
        shot = self.data.get("shot_scale", {}).get("experiments", {})
        movement = self.data.get("movement", {}).get("experiments", {})

        cards = []
        for collection, entries, detail in (
            (shot, (("baseline_cnn", "Shot-scale baseline"),
                    ("augmented_cnn", "Shot-scale augmented")),
             "Test accuracy | shot-scale task"),
            (movement, (("baseline", "Movement baseline"),
                        ("balanced", "Movement balanced")),
             "Test accuracy | original movement labels"),
        ):
            for key, label in entries:
                metrics = collection.get(key, {}).get("metrics", {})
                cards.append((
                    label, format_percent(metrics.get("test_accuracy")), detail
                ))
        if cards:
            self._cards(body, cards, columns=2)

        self._section(
            body, "Model performance",
            "Compare saved test metrics. Bar lengths use the metric value on a 0 to 100% scale."        
            )
        for task_title, collection, specs in (
            ("Shot-scale accuracy", shot, (
                ("baseline_cnn", "Baseline CNN", COLORS["accent"]),
                ("augmented_cnn", "Augmented CNN", COLORS["cyan"]),
            )),
            ("Camera-movement accuracy", movement, (
                ("baseline", "Baseline CNN", COLORS["accent"]),
                ("balanced", "Balanced-sampling CNN", COLORS["cyan"]),
            )),
        ):
            entries = [
                (label, collection.get(key, {}).get("metrics", {}), color)
                for key, label, color in specs
                if collection.get(key, {}).get("metrics")
            ]
            if entries:
                self._comparison_chart(
                    body, task_title, entries, "test_accuracy",
                    "Test accuracy",
                    "Saved evaluation result | higher accuracy does not guarantee balanced class performance"
                )

        self._section(body, "Experiment overview")
        rows = self._comparison_rows()
        if rows:
            self._tree(
                body, ("task", "experiment", "accuracy", "macro_f1",
                       "weighted_f1", "samples"), rows,
                ("Task", "Experiment", "Test accuracy", "Macro F1",
                 "Weighted F1", "Test samples"), height=4
            )
        else:
            self._empty_state(body, "No usable experiment summaries are available.")

        baseline = movement.get("baseline", {}).get("metrics", {})
        balanced = movement.get("balanced", {}).get("metrics", {})
        base_acc = baseline.get("test_accuracy")
        balanced_acc = balanced.get("test_accuracy")
        base_f1 = baseline.get("macro_f1")
        balanced_f1 = balanced.get("macro_f1")
        self._section(body, "Experimental finding")
        insight = self._panel(body, padding=16)
        insight.pack(fill="x", pady=7)
        tk.Label(
            insight, text="READ THE RESULTS, NOT JUST THE ACCURACY",
            bg=COLORS["panel"], fg=COLORS["cyan"],
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")
        if all(isinstance(v, (int, float)) for v in
               (base_acc, balanced_acc, base_f1, balanced_f1)):
            summary = (
                f"On the saved movement test set, baseline accuracy is "
                f"{format_percent(base_acc)} versus {format_percent(balanced_acc)} "
                f"for balanced sampling. Macro F1 is "
                f"{format_percent(base_f1)} versus {format_percent(balanced_f1)}. "
                "These saved results do not show an improvement from balanced sampling."
            )
        else:
            summary = (
                "Movement comparisons are incomplete because one or more saved "
                "summary metrics are unavailable."
            )
        tk.Label(
            insight, text=summary, bg=COLORS["panel"],
            fg=COLORS["text"], font=("Segoe UI", 10),
            wraplength=1000, justify="left"
        ).pack(anchor="w", pady=(7, 0))

        self._section(body, "Label scope")
        tk.Label(
            body,
            text=("Shot scale: CS, ECS, FS, LS, MS.  |  Original movement labels: "
                  "Static, Motion, Push, Pull, Multi_movement. The movement model "
                  "is not a validated Pan/Tilt/Zoom/Tracking-Dolly classifier."),
            bg=COLORS["panel"], fg=COLORS["muted"], padx=14, pady=13,
            font=("Segoe UI", 9), wraplength=1000, justify="left"
        ).pack(fill="x", pady=6)
        self._warnings(body, self.data.get("warnings", []))

    def _navigate_highlight(self, page):
        for name, button in self._nav_buttons.items():
            selected = name == page
            button.configure(
                bg=COLORS["selected"] if selected else COLORS["sidebar"],
                fg=COLORS["accent"] if selected else COLORS["muted"],
                font=("Segoe UI", 10, "bold" if selected else "normal"),
            )

    def show_explorer(self, task):
        self._navigate_highlight(task)
        self._clear()
        is_shot = task == "shot_scale"
        title = ("Shot-Scale Evaluation Explorer" if is_shot
                 else "Camera-Movement Evaluation Explorer")
        subtitle = (
            "Classes: CS, ECS, FS, LS, MS  |  Saved test evaluations"
            if is_shot else
            "Original MovieShots classes only  |  Not Pan/Tilt/Zoom/Tracking-Dolly"
        )
        self._header(title, subtitle, "CINESCOPE AI  /  EVALUATION")
        body = self._scroll_area()
        group = self.data.get("shot_scale" if is_shot else "movement", {})
        collection = group.get("experiments", {})
        available = [
            (key, item) for key, item in collection.items()
            if isinstance(item, dict) and (
                item.get("metrics") or item.get("per_class_metrics")
            )
        ]
        if not available:
            self._empty_state(body, "No usable evaluation artifacts were found for this task.")
            self._warnings(body, self.data.get("warnings", []))
            return

        for key, experiment in available:
            label = experiment.get("label", key)
            self._section(body, label)
            metrics = experiment.get("metrics", {})
            self._cards(body, [
                ("Test accuracy", format_percent(metrics.get("test_accuracy")),
                 "Saved test evaluation"),
                ("Macro F1", format_percent(metrics.get("macro_f1")),
                 "Equal weight per class"),
                ("Weighted F1", format_percent(metrics.get("weighted_f1")),
                 "Weighted by class support"),
                ("Test samples", str(metrics.get(
                    "test_samples", metrics.get("test_frames", "?")
                )), "Saved evaluation metadata"),
            ], columns=2)

            per_class = experiment.get("per_class_metrics", {})
            if isinstance(per_class, dict) and per_class:
                rows = []
                for class_name, values in per_class.items():
                    if not isinstance(values, dict):
                        continue
                    rows.append((
                        class_name,
                        format_percent(values.get("precision")),
                        format_percent(values.get("recall")),
                        format_percent(values.get("f1-score", values.get("f1_score"))),
                        values.get("support", "?"),
                    ))
                if rows:
                    self._section(
                        body, "Class-wise performance",
                        "Precision, recall and F1 are shown as percentages; support is the saved class count."
                    )
                    self._tree(
                        body, ("class", "precision", "recall", "f1", "support"),
                        rows, ("Class", "Precision", "Recall", "F1 score", "Support"),
                        height=min(8, max(3, len(rows)))
                    )

            matrix = experiment.get("confusion_matrix")
            labels = experiment.get("confusion_matrix_labels_order")
            if matrix and labels:
                self._section(
                    body, "Confusion matrix",
                    "Rows = actual class  |  Columns = predicted class  |  Exact saved counts"
                )
                self._matrix(body, labels, matrix)

        self._warnings(body, self.data.get("warnings", []))

    def _matrix(self, parent, labels, matrix):
        if (not isinstance(matrix, list) or len(matrix) != len(labels)
                or any(not isinstance(row, (list, tuple))
                       or len(row) != len(labels) for row in matrix)):
            self._warning_line(
                parent, "Confusion matrix dimensions do not match the class labels."
            )
            return

        maximum = max(
            (value for row in matrix for value in row
             if isinstance(value, (int, float))), default=0
        )
        count = len(labels)
        cell_w = 78 if count <= 5 else 62
        row_h = 43
        left_w = 128
        top_h = 76
        width = left_w + count * cell_w + 12
        height = top_h + count * row_h + 12
        outer = self._panel(parent, padding=10)
        outer.pack(fill="x", pady=7)
        canvas = tk.Canvas(
            outer, height=min(height, 360), bg=COLORS["panel"],
            highlightthickness=0, bd=0
        )
        canvas.pack(fill="x", expand=True)
        canvas.configure(scrollregion=(0, 0, width, height))

        def draw():
            canvas.delete("all")
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(
                left_w + count * cell_w / 2, 10,
                text="PREDICTED CLASS", anchor="n",
                fill=COLORS["muted"], font=("Segoe UI", 8, "bold")
            )
            for col, name in enumerate(labels):
                x = left_w + col * cell_w + cell_w / 2
                canvas.create_text(
                    x, 37, text=str(name), fill=COLORS["cyan"],
                    font=("Segoe UI", 8, "bold"), width=cell_w - 5
                )
            canvas.create_text(
                8, top_h + count * row_h / 2,
                text="ACTUAL", anchor="w", angle=90,
                fill=COLORS["muted"], font=("Segoe UI", 8, "bold")
            )
            for row_index, row in enumerate(matrix):
                y = top_h + row_index * row_h
                canvas.create_text(
                    left_w - 12, y + row_h / 2,
                    text=str(labels[row_index]), anchor="e",
                    fill=COLORS["text"], font=("Segoe UI", 9)
                )
                for col_index, value in enumerate(row):
                    x = left_w + col_index * cell_w
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = 0
                    intensity = numeric / maximum if maximum else 0
                    if row_index == col_index:
                        base = (48, 77, 91)
                        bright = (80, 183, 170)
                    else:
                        base = (36, 42, 58)
                        bright = (117, 95, 172)
                    rgb = tuple(
                        int(a + (b - a) * intensity)
                        for a, b in zip(base, bright)
                    )
                    fill = "#{:02x}{:02x}{:02x}".format(*rgb)
                    canvas.create_rectangle(
                        x + 2, y + 2, x + cell_w - 2, y + row_h - 2,
                        fill=fill, outline=COLORS["border"]
                    )
                    foreground = "#FFFFFF" if intensity > 0.45 else COLORS["text"]
                    canvas.create_text(
                        x + cell_w / 2, y + row_h / 2,
                        text=str(value), fill=foreground,
                        font=("Segoe UI", 10, "bold" if row_index == col_index else "normal")
                    )
            canvas.configure(scrollregion=(0, 0, width, height))

        draw()
        canvas.bind("<Configure>", lambda _event: draw())

    def _warning_line(self, parent, text):
        tk.Label(
            parent, text="!  " + text, bg=COLORS["warning_bg"],
            fg=COLORS["warning"], font=("Segoe UI", 9),
            wraplength=950, justify="left", padx=12, pady=10
        ).pack(fill="x", pady=5)

    def _empty_state(self, parent, text):
        panel = self._panel(parent, padding=20)
        panel.pack(fill="x", pady=12)
        tk.Label(
            panel, text="NO EVALUATION DATA", bg=COLORS["panel"],
            fg=COLORS["accent"], font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")
        tk.Label(
            panel, text=text, bg=COLORS["panel"],
            fg=COLORS["muted"], font=("Segoe UI", 10),
            wraplength=900, justify="left"
        ).pack(anchor="w", pady=(8, 0))

    def show_status(self):
        self._navigate_highlight("status")
        self._clear()
        self._header(
            "Evaluation Data Status",
            "Artifact availability and validation warnings  |  Read-only inspection",
            "CINESCOPE AI  /  SYSTEM STATUS"
        )
        body = self._scroll_area()
        self._section(
            body, "Loaded artifact groups",
            "Status reflects the saved evaluation artifacts read by the existing data loader."
        )
        rows = []
        for task in ("shot_scale", "movement"):
            group = self.data.get(task, {})
            experiments = group.get("experiments", {})
            rows.append((
                task.replace("_", " ").title(),
                group.get("source", "?"),
                "Available" if experiments else "Missing / invalid",
                len(experiments),
            ))
        self._tree(
            body, ("task", "source", "status", "experiments"), rows,
            ("Task", "Artifact source", "Status", "Experiments"), height=2
        )

        warnings = self.data.get("warnings", [])
        self._section(body, "Validation result")
        if warnings:
            panel = self._panel(body, padding=15)
            panel.pack(fill="x", pady=7)
            tk.Label(
                panel, text=f"{len(warnings)} NOTICE(S) REQUIRE REVIEW",
                bg=COLORS["panel"], fg=COLORS["warning"],
                font=("Segoe UI", 10, "bold")
            ).pack(anchor="w")
            self._warnings(body, warnings)
        else:
            panel = tk.Frame(
                body, bg=COLORS["success_bg"],
                highlightbackground=COLORS["success"], highlightthickness=1,
                padx=16, pady=15
            )
            panel.pack(fill="x", pady=7)
            tk.Label(
                panel, text="?  VALIDATION PASSED",
                bg=COLORS["success_bg"], fg=COLORS["success"],
                font=("Segoe UI", 10, "bold")
            ).pack(anchor="w")
            tk.Label(
                panel,
                text="No missing or inconsistent artifacts were detected in the files checked.",
                bg=COLORS["success_bg"], fg=COLORS["text"],
                font=("Segoe UI", 9), wraplength=950, justify="left"
            ).pack(anchor="w", pady=(6, 0))

        self._section(body, "Scope and safeguards")
        tk.Label(
            body,
            text=("This workspace displays saved evaluation results only. It does not train "
                  "models or run inference. The movement labels remain Static, Motion, Push, "
                  "Pull, and Multi_movement; they are not a validated Pan/Tilt/Zoom/Tracking-Dolly mapping."),
            bg=COLORS["panel"], fg=COLORS["muted"],
            padx=14, pady=13, font=("Segoe UI", 9),
            wraplength=1000, justify="left"
        ).pack(fill="x", pady=7)


if __name__ == "__main__":
    try:
        app = CineScopeApp()
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("CineScope AI startup error", str(exc))
        raise
