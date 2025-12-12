import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import copy
import json
import math
import os
import sys

from PIL import Image, ImageTk

try:
    from tkintermapview import TkinterMapView
except ImportError:  # pragma: no cover - dependency issue surfaced at runtime
    TkinterMapView = None

from graph import load_graph
from topk import yen_k_shortest_paths
from algorithms.cus1_ucs import cus1_ucs
from algorithms.dfs import dfs_search
from algorithms.bfs import bfs_search
from algorithms.gbfs import gbfs_search
from algorithms.astar import astar_search
from algorithms.cus2_hcs import cus2_hcs
from predictor import predict_severity

# =============================
# ICS GUI (Assignment 2B)
# =============================
class ICS_GUI:
    SEVERITY_LEVELS = {
        "Minor": 1.15,
        "Moderate": 1.50,
        "None": 1.00,
        "Severe": 2.00,
    }
    MAX_ACCIDENTS = 3
    ORIGIN_PLACEHOLDER = "-- choose start --"
    TARGET_PLACEHOLDER = "-- choose end --"
    CANVAS_WIDTH = 760
    CANVAS_HEIGHT = 700
    MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")

    def __init__(self, root):
        self.root = root
        self.root.title("ICS – Incident Classification System")
        self.root.geometry("1350x750")

        if TkinterMapView is None:
            messagebox.showerror(
                "Missing dependency",
                "tkintermapview is required. Install it with 'pip install tkintermapview'.",
            )
            raise SystemExit(1)

        self.graph = {}
        self.graph_original = {}
        self.origin_default = None
        self.destination_defaults = []
        self.coords = {}
        self.accident = {}
        self.user_accidents = {}
        self.landmarks = {}
        self.name_to_id = {}
        self.current_paths = []
        self.map_widget = None
        self.map_markers = []
        self.route_paths = []
        self.current_meta_zoom = 15
        self.edge_polylines = {}
        self.routes_stale = True
        self.routing_status_var = tk.StringVar(value="Routes not yet computed.")
        self.animation_after_id = None
        self.animation_path_handle = None
        self.animation_points = []
        self.animation_step = 0
        self.base_edge_handles = []

        self.model_options = ["CNN", "MobileNetV2", "Random Forest"]
        self.selected_model_var = tk.StringVar(value=self.model_options[0])
        self.severity_choices = list(self.SEVERITY_LEVELS.keys())

        self.map_entries = self.discover_map_files()
        if not self.map_entries:
            messagebox.showerror(
                "Missing maps",
                "No Kuching map files were found in the maps/ directory.",
            )
            self.map_entries = [
                {
                    "label": "No maps available",
                    "map_path": None,
                    "image_path": None,
                    "meta": None,
                }
            ]

        self.map_background_image = None
        self.background_offset = (0, 0)
        self.background_display_size = (self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        self.current_extent = None
        self.current_projection = None
        self.current_routes = []
        self.active_route_index = 0
        self.last_origin_id = None
        self.last_destination_id = None
        self.edge_polylines = {}

        self.current_map_entry = self.map_entries[0]
        self.map_var = tk.StringVar(value=self.current_map_entry["label"])
        self.map_label_var = tk.StringVar()
        self.accident_entries = []
        self.accident_entries_container = None
        self.no_accidents_label = None
        self.add_accident_button = None
        self.model_run_summary_var = tk.StringVar(value="No accident predictions yet.")
        self.model_label = None
        self.model_info_label = None
        self.final_label = None
        self.run_routing_button = None
        self._suspend_entry_sync = False
        self.k_var = tk.StringVar(value="3")
        self.route_cards_container = None

        if self.current_map_entry.get("map_path"):
            self.load_map_data(self.current_map_entry)
        else:
            self.graph = {}
            self.landmarks = {}
            self.map_label_var.set("Map: (missing)")

        self.build_layout()
        self.draw_map()

    # ---------------------------
    # Main UI layout
    # ---------------------------
    def build_layout(self):

    # --- Main Window Size ---
        self.root.geometry("1280x720")   # slightly smaller, fits screens better

        # ============================
        # LEFT PANEL (Map)
        # ============================
        left = tk.Frame(self.root, width=800, height=700, bg="white")
        left.pack(side="left", fill="both", expand=True)

        self.map_widget = TkinterMapView(
            left,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            corner_radius=0,
        )
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_zoom(14)
        self.map_widget.set_position(1.557, 110.343)

        # ============================
        # RIGHT PANEL (SCROLLABLE)
        # ============================

        # Outer frame holding canvas + scrollbar
        right_outer = tk.Frame(self.root, width=480, bg="#f5f5f5")
        right_outer.pack(side="right", fill="y")

        # Canvas used for scrolling
        right_canvas = tk.Canvas(right_outer, bg="#f5f5f5", width=480, highlightthickness=0)
        right_canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        right_canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            if event.delta:
                right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        right_canvas.bind("<Enter>", lambda _e: right_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        right_canvas.bind("<Leave>", lambda _e: right_canvas.unbind_all("<MouseWheel>"))

        # Actual content frame placed INSIDE the canvas
        right_container = tk.Frame(right_canvas, bg="#f5f5f5")
        container_window = right_canvas.create_window((0, 0), window=right_container, anchor="nw")

        def _sync_scrollregion(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            right_canvas.itemconfig(container_window, width=right_canvas.winfo_width())

        right_container.bind("<Configure>", _sync_scrollregion)

        def _init_scroll():
            _sync_scrollregion(None)
            right_canvas.yview_moveto(0)

        self.root.after(50, _init_scroll)

        right = tk.Frame(right_container, bg="#f5f5f5", padx=30, pady=30)
        right.pack(expand=True)

        # ==============================
        # Accident Detection & Impact
        # ==============================
        incidents = tk.LabelFrame(
            right,
            text="Accident Detection & Impact",
            bg="#f5f5f5",
            padx=12,
            pady=12,
        )
        incidents.pack(fill="x", pady=10)

        model_row = tk.Frame(incidents, bg="#f5f5f5")
        model_row.pack(fill="x", pady=(0, 6))

        tk.Label(model_row, text="Model:", bg="#f5f5f5").pack(side="left")
        model_menu = tk.OptionMenu(model_row, self.selected_model_var, *self.model_options)
        model_menu.config(width=20)
        model_menu.pack(side="left", padx=(6, 12))

        tk.Button(
            model_row,
            text="Run Model",
            command=self.run_ml_prediction,
            width=12,
        ).pack(side="right")

        self.model_summary_label = tk.Label(
            incidents,
            textvariable=self.model_run_summary_var,
            anchor="w",
            bg="#f5f5f5",
            fg="#444",
        )
        self.model_summary_label.pack(fill="x", pady=(0, 4))

        self.model_label = tk.Label(
            incidents,
            text="Primary model output: -",
            anchor="w",
            bg="#f5f5f5",
            fg="#222",
        )
        self.model_label.pack(fill="x")

        self.model_info_label = tk.Label(
            incidents,
            text="Notes: -",
            anchor="w",
            bg="#f5f5f5",
            fg="#555",
        )
        self.model_info_label.pack(fill="x", pady=(0, 2))

        self.final_label = tk.Label(
            incidents,
            text="Applied severity levels: (none)",
            anchor="w",
            bg="#f5f5f5",
            fg="darkred",
            font=("Arial", 11, "bold"),
        )
        self.final_label.pack(fill="x", pady=(0, 6))

        tk.Label(
            incidents,
            text="Add up to three accident reports. Each report selects a road segment and image.",
            wraplength=360,
            justify="left",
            bg="#f5f5f5",
            fg="#555",
        ).pack(fill="x")

        self.accident_entries_container = tk.Frame(incidents, bg="#f5f5f5")
        self.accident_entries_container.pack(fill="both", expand=True, pady=(8, 0))

        self.no_accidents_label = tk.Label(
            self.accident_entries_container,
            text="No accidents added. Click 'Add Accident' to begin.",
            bg="#f5f5f5",
            fg="#666",
            anchor="w",
        )
        self.no_accidents_label.pack(fill="x", pady=6)

        self.add_accident_button = tk.Button(
            incidents,
            text="+ Add Accident",
            command=self.add_accident_entry,
            width=20,
        )
        self.add_accident_button.pack(pady=(10, 0))

        # ==============================
        # Routing Controls
        # ==============================
        tk.Label(right, text="Route Finder", font=("Arial", 16, "bold"), bg="#f5f5f5").pack(pady=(20, 0))

        routing_wrapper = tk.Frame(right, bg="#f5f5f5")
        routing_wrapper.pack(fill="x", pady=10, padx=20)

        def make_row(parent, label_text, widget_builder, row_idx):
            tk.Label(parent, text=label_text, anchor="w", bg="#f5f5f5").grid(row=row_idx, column=0, sticky="w", pady=(4, 0))
            widget = widget_builder(parent)
            widget.grid(row=row_idx, column=1, sticky="ew", pady=(4, 0))
            return widget

        selections = tk.Frame(routing_wrapper, bg="#f5f5f5")
        selections.pack(fill="x")

        map_labels = [entry["label"] for entry in self.map_entries]
        self.map_menu = make_row(
            selections,
            "Map:",
            lambda parent: tk.OptionMenu(parent, self.map_var, *map_labels, command=self.on_map_change),
            0,
        )

        tk.Label(selections, text="Origin:", anchor="w",
                bg="#f5f5f5").grid(row=1, column=0, sticky="w", pady=(8, 0))
        origin_names = self.landmark_names() or ["(none)"]
        self.origin_var = tk.StringVar(value="-- choose origin --" if origin_names else "(none)")
        self.origin_menu = tk.OptionMenu(selections, self.origin_var, "-- choose origin --", *origin_names)
        self.origin_menu.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        tk.Label(selections, text="Destination:", anchor="w",
                bg="#f5f5f5").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.destination_var = tk.StringVar(value="-- choose destination --" if origin_names else "(none)")
        self.destination_menu = tk.OptionMenu(selections, self.destination_var, "-- choose destination --", *origin_names)
        self.destination_menu.grid(row=2, column=1, sticky="ew", pady=(8, 0))

        tk.Label(selections, text="Algorithm:", anchor="w", bg="#f5f5f5").grid(row=3, column=0, sticky="w", pady=(8, 0))
        algorithm_options = [
            ("CUS1", "Uniform Cost Search (CUS1)"),
            ("DFS", "Depth-First Search"),
            ("BFS", "Breadth-First Search"),
            ("GBFS", "Greedy Best-First Search"),
            ("ASTAR", "A* Search"),
            ("CUS2", "Hill Climbing Search (CUS2)"),
        ]
        self.algorithm_var = tk.StringVar(value=algorithm_options[0][1])
        self.algorithm_menu = tk.OptionMenu(
            selections,
            self.algorithm_var,
            *[label for _, label in algorithm_options],
        )
        self.algorithm_key_lookup = {label: key for key, label in algorithm_options}
        self.algorithm_menu.grid(row=3, column=1, sticky="ew", pady=(8, 0))

        selections.grid_columnconfigure(1, weight=1)

        routes_section = tk.LabelFrame(
            right,
            text="Routes",
            bg="#f5f5f5",
            padx=10,
            pady=10,
        )
        routes_section.pack(fill="x", pady=10, padx=10)

        self.route_cards_container = tk.Frame(routes_section, bg="#f5f5f5")
        self.route_cards_container.pack(fill="x", expand=True)
        self.render_route_cards()

        self.run_routing_button = tk.Button(
            right,
            text="Run Routing",
            command=self.run_routing,
            width=25,
        )
        self.run_routing_button.pack(pady=10)
        tk.Label(
            right,
            textvariable=self.routing_status_var,
            fg="#8a2d0c",
            bg="#f5f5f5",
            wraplength=360,
            justify="left",
        ).pack(pady=(0, 10), padx=10, fill="x")

        tk.Label(right, textvariable=self.map_label_var, fg="#555", bg="#f5f5f5").pack()

        # Output box with scroll
        output_frame = tk.Frame(right, bg="#f5f5f5")
        output_frame.pack(fill="both", expand=True, pady=(10, 0))

        scroll = tk.Scrollbar(output_frame, orient="vertical")
        scroll.pack(side="right", fill="y")

        self.route_output = tk.Text(output_frame, height=10, width=40, yscrollcommand=scroll.set, wrap="word")
        self.route_output.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.route_output.yview)


    # ---------------------------
    # ML Prediction Pipeline
    # ---------------------------
    def run_ml_prediction(self):
        active_entries = [entry for entry in self.accident_entries if entry.get("image_path")]
        if not active_entries:
            messagebox.showwarning("No images", "Add at least one accident image before running the model.")
            return

        selected_model_raw = self.selected_model_var.get()
        model_key = selected_model_raw.lower()

        # Normalize names
        if "mobile" in model_key:
            model_key = "mobilenetv2"
        elif "cnn" in model_key:
            model_key = "cnn"
        elif "forest" in model_key:
            model_key = "random forest"


        applied = []
        last_note = ""
        self._suspend_entry_sync = True
        try:
            for entry in active_entries:
                try:
                    result = self._predict_severity(entry["image_path"], model_key)
                except Exception as exc:
                    messagebox.showerror("Prediction error", f"Failed to process image:\n{exc}")
                    return
                severity = result["severity"]
                multiplier = result["multiplier"]
                note_text = result["note"]
                entry["multiplier_dirty"] = False
                entry["severity_var"].set(severity)
                self._set_entry_multiplier(entry, multiplier, reset_dirty=True)
                entry["note_var"].set(note_text)
                entry["status_var"].set(note_text)
                applied.append(severity)
                last_note = note_text
        finally:
            self._suspend_entry_sync = False

        if applied:
            self.model_label.config(text=f"Model Used: {selected_model_raw}")
            self.model_info_label.config(text=f"Notes: {last_note or 'Model applied.'}")
            self.final_label.config(text=f"Applied severity levels: {', '.join(applied)}")
        self.sync_accidents_to_graph()
        self.model_run_summary_var.set(
            f"{selected_model_raw} processed {len(active_entries)} accident(s)."
        )
        self.mark_routes_stale("New severity predictions applied.")

    def _predict_severity(self, image_path, model_name):
        severity = predict_severity(image_path, model_name)

        # Safety check (in case model returns something unexpected)
        if severity not in self.SEVERITY_LEVELS:
            severity = "Moderate"

        multiplier = self.SEVERITY_LEVELS[severity]

        note = f"Predicted as {severity} using {model_name}"
        return {"severity": severity, "multiplier": multiplier, "note": note}

    # ---------------------------
    # Accident Entry Management
    # ---------------------------
    def add_accident_entry(self):
        if self.accident_entries_container is None:
            return
        if len(self.accident_entries) >= self.MAX_ACCIDENTS:
            messagebox.showinfo("Limit reached", "You can only track up to three accidents at a time.")
            return

        card = tk.Frame(self.accident_entries_container, bg="white", bd=1, relief="solid")
        card.pack(fill="x", pady=6)

        header = tk.Frame(card, bg="white")
        header.pack(fill="x")
        title_label = tk.Label(header, text="", font=("Arial", 12, "bold"), bg="white")
        title_label.pack(side="left")
        remove_btn = tk.Button(header, text="Remove", command=lambda frame=card: self.remove_accident_entry(frame))
        remove_btn.pack(side="right")

        # Variables
        origin_var = tk.StringVar(value=self.ORIGIN_PLACEHOLDER)
        target_var = tk.StringVar(value=self.TARGET_PLACEHOLDER)
        severity_var = tk.StringVar(value="Pending")
        multiplier_var = tk.DoubleVar(value=1.00)
        note_var = tk.StringVar(value="Upload an image to classify.")
        status_var = tk.StringVar(value="Upload an image to classify.")

        # Origin and Target Rows
        row1 = tk.Frame(card, bg="white")
        row1.pack(fill="x", pady=(6, 2))
        tk.Label(row1, text="Starts at:", bg="white").grid(row=0, column=0, sticky="w")
        origin_menu = tk.OptionMenu(row1, origin_var, self.ORIGIN_PLACEHOLDER)
        origin_menu.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        row1.grid_columnconfigure(1, weight=1)

        row2 = tk.Frame(card, bg="white")
        row2.pack(fill="x", pady=(2, 6))
        tk.Label(row2, text="Ends at:", bg="white").grid(row=0, column=0, sticky="w")
        target_menu = tk.OptionMenu(row2, target_var, self.TARGET_PLACEHOLDER)
        target_menu.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        row2.grid_columnconfigure(1, weight=1)

        # Upload Image
        upload_row = tk.Frame(card, bg="white")
        upload_row.pack(fill="x", pady=(0, 6))
        tk.Button(upload_row, text="Upload Image", command=lambda entry_frame=card: self.upload_entry_image(entry_frame), width=15).pack(side="left")
        image_label = tk.Label(upload_row, text="No image", bg="#eeeeee", width=25, height=5)
        image_label.pack(side="left", padx=10)

        # AI-Controlled Severity + Multiplier Row
        severity_row = tk.Frame(card, bg="white")
        severity_row.pack(fill="x", pady=(0, 4))

        tk.Label(severity_row, text="Severity:", bg="white").grid(row=0, column=0, sticky="w")
        severity_label = tk.Label(severity_row, textvariable=severity_var, bg="white", fg="darkred")
        severity_label.grid(row=0, column=1, sticky="w", padx=(6, 8))

        tk.Label(severity_row, text="Multiplier:", bg="white").grid(row=0, column=2, sticky="w")
        multiplier_label = tk.Label(severity_row, textvariable=multiplier_var, bg="white")
        multiplier_label.grid(row=0, column=3, sticky="w", padx=(6, 0))

        # Formula + Status
        formula_var = tk.StringVar(value="Formula: time × Pending × 1.00")
        formula_label = tk.Label(card, textvariable=formula_var, bg="white", fg="#333", anchor="w")
        formula_label.pack(fill="x", pady=(0, 4))

        status_label = tk.Label(card, textvariable=status_var, anchor="w", bg="white", fg="#555", wraplength=360, justify="left")
        status_label.pack(fill="x", pady=(0, 4))

        # Track Entry
        entry = {
            "frame": card,
            "title_label": title_label,
            "origin_var": origin_var,
            "target_var": target_var,
            "origin_menu": origin_menu,
            "target_menu": target_menu,
            "image_label": image_label,
            "image_thumb": None,
            "image_path": None,
            "severity_var": severity_var,
            "severity_label": severity_label,
            "note_var": note_var,
            "status_var": status_var,
            "multiplier_var": multiplier_var,
            "formula_var": formula_var,
            "multiplier_dirty": False,
            "suppress_multiplier_trace": False,
        }

        origin_var.trace_add("write", lambda *_args, e=entry: self.on_accident_entry_changed(e))
        target_var.trace_add("write", lambda *_args, e=entry: self.on_accident_entry_changed(e))

        self.accident_entries.append(entry)
        self.populate_entry_dropdowns(entry)
        self.update_entry_severity_display(entry)
        self.update_accident_cards_header()
        self.ensure_no_accidents_label()
        self.update_add_accident_button_state()
        self.sync_accidents_to_graph()


    def remove_accident_entry(self, frame):
        target_entry = None
        for entry in self.accident_entries:
            if entry["frame"] == frame:
                target_entry = entry
                break
        if not target_entry:
            return
        target_entry["frame"].destroy()
        self.accident_entries.remove(target_entry)
        self.update_accident_cards_header()
        self.ensure_no_accidents_label()
        self.update_add_accident_button_state()
        self.sync_accidents_to_graph()

    def ensure_no_accidents_label(self):
        if self.no_accidents_label is None:
            return
        if self.accident_entries:
            if self.no_accidents_label.winfo_manager():
                self.no_accidents_label.pack_forget()
        else:
            self.no_accidents_label.pack(fill="x", pady=6)

    def update_accident_cards_header(self):
        for idx, entry in enumerate(self.accident_entries, start=1):
            entry["title_label"].config(text=f"Accident {idx}")

    def update_add_accident_button_state(self):
        if self.add_accident_button is None:
            return
        if len(self.accident_entries) >= self.MAX_ACCIDENTS:
            self.add_accident_button.config(state="disabled", text="Maximum accidents added")
        else:
            self.add_accident_button.config(state="normal", text="+ Add Accident")

    def entry_landmark_choices(self):
        names = self.landmark_names()
        if names:
            return names
        return []

    def populate_entry_dropdowns(self, entry):
        choices = self.entry_landmark_choices()
        origin_menu = entry["origin_menu"]["menu"]
        target_menu = entry["target_menu"]["menu"]

        def rebuild(menu, var, placeholder):
            menu.delete(0, "end")
            if not choices:
                menu.add_command(label="(no landmarks)", command=lambda: None)
                var.set("(no landmarks)")
                return
            menu.add_command(label=placeholder, command=lambda v=placeholder: var.set(v))
            for name in choices:
                menu.add_command(label=name, command=lambda v=name: var.set(v))
            if var.get() not in ([placeholder] + choices):
                var.set(placeholder)

        rebuild(origin_menu, entry["origin_var"], self.ORIGIN_PLACEHOLDER)
        rebuild(target_menu, entry["target_var"], self.TARGET_PLACEHOLDER)

    def refresh_accident_entry_menus(self):
        for entry in self.accident_entries:
            self.populate_entry_dropdowns(entry)

    def on_accident_entry_changed(self, entry):
        entry["severity_var"].set("Pending")
        entry["status_var"].set("Waiting for classification.")
        self.sync_accidents_to_graph()

    def update_entry_severity_display(self, entry):
        severity = entry["severity_var"].get()
        entry["severity_label"].config(text=f"Severity: {severity}")
        if severity in self.SEVERITY_LEVELS and not entry.get("multiplier_dirty"):
            self._set_entry_multiplier(entry, self.SEVERITY_LEVELS[severity], reset_dirty=False)
        self.update_entry_formula(entry)
        if not self._suspend_entry_sync:
            self.sync_accidents_to_graph()

    def _set_entry_multiplier(self, entry, value, reset_dirty=True):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 1.0
        entry["suppress_multiplier_trace"] = True
        entry["multiplier_var"].set(round(numeric, 2))
        entry["suppress_multiplier_trace"] = False
        if reset_dirty:
            entry["multiplier_dirty"] = False
        self.update_entry_formula(entry)

    def update_entry_formula(self, entry):
        severity = entry["severity_var"].get()
        try:
            multiplier = float(entry["multiplier_var"].get())
        except (tk.TclError, ValueError, TypeError):
            multiplier = 1.0
        entry["formula_var"].set(f"Formula: time × {severity} × {multiplier:.2f}")

    def on_entry_multiplier_changed(self, entry, *_args):
        if entry.get("suppress_multiplier_trace"):
            return
        entry["multiplier_dirty"] = True
        self.update_entry_formula(entry)
        self.sync_accidents_to_graph()

    def upload_entry_image(self, frame):
        entry = next((item for item in self.accident_entries if item["frame"] == frame), None)
        if entry is None:
            return
        filename = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not filename:
            return
        try:
            image = Image.open(filename)
        except Exception:
            messagebox.showerror("Image error", "Unable to open the selected image.")
            return
        thumbnail = image.copy()
        thumbnail.thumbnail((220, 180))
        photo = ImageTk.PhotoImage(thumbnail)
        entry["image_thumb"] = photo
        entry["image_label"].config(image=photo, text="")
        entry["image_path"] = filename
        entry["severity_var"].set("Pending")
        entry["status_var"].set("Image ready. Run the model to classify.")
        self.model_run_summary_var.set("Images updated. Run the model to classify.")
        self.sync_accidents_to_graph()

    def clear_all_accident_entries(self):
        if not hasattr(self, "accident_entries") or not self.accident_entries:
            if self.no_accidents_label and not self.no_accidents_label.winfo_manager():
                self.no_accidents_label.pack(fill="x", pady=6)
            return
        for entry in list(self.accident_entries):
            entry["frame"].destroy()
        self.accident_entries.clear()
        self.ensure_no_accidents_label()
        self.update_add_accident_button_state()
        self.sync_accidents_to_graph()

    def sync_accidents_to_graph(self):
        if not hasattr(self, "user_accidents"):
            return
        self.user_accidents = {}
        for entry in self.accident_entries:
            origin_name = entry["origin_var"].get()
            target_name = entry["target_var"].get()
            if (
                not origin_name
                or not target_name
                or origin_name.startswith("--")
                or target_name.startswith("--")
                or origin_name.startswith("(")
                or target_name.startswith("(")
            ):
                continue
            try:
                origin_id = self.resolve_node(origin_name)
                target_id = self.resolve_node(target_name)
            except ValueError:
                continue
            severity = entry["severity_var"].get()
            try:
                multiplier = float(entry["multiplier_var"].get())
            except (tk.TclError, ValueError, TypeError):
                multiplier = self.SEVERITY_LEVELS.get(severity, 1.0)
            formula_var = entry.get("formula_var")
            formula_text = formula_var.get() if formula_var else ""
            self.user_accidents[(origin_id, target_id)] = {
                "severity": severity,
                "multiplier": multiplier,
                "image_path": entry.get("image_path"),
                "model": self.selected_model_var.get(),
                "formula": formula_text,
            }
        self.rebuild_graph_with_accidents()
        if self.user_accidents:
            self.mark_routes_stale("Accident slowdowns updated.")
            if not getattr(self, "_auto_update_lock", False):
                self._auto_update_lock = True
                self.run_routing()
                self._auto_update_lock = False

    # ---------------------------
    # Routing
    # ---------------------------
    def load_map_data(self, entry):
        map_path = entry.get("map_path")
        if not map_path:
            self.graph = {}
            self.landmarks = {}
            self.map_label_var.set("Map: (missing)")
            return
        try:
            (
                self.graph,
                self.origin_default,
                self.destination_defaults,
                self.coords,
                self.accident,
                self.landmarks,
            ) = load_graph(map_path)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Map file not found: {map_path}")
            self.graph = {}
            self.landmarks = {}
            return

        self.current_map_entry = entry
        self.current_map_path = map_path
        friendly = entry.get("label", os.path.basename(map_path))
        self.map_label_var.set(f"Map: {friendly}")

        self.load_background_assets(entry)
        self.name_to_id = {name: node for node, name in self.landmarks.items()}
        if hasattr(self, "origin_menu"):
            self.refresh_landmark_menus()
        meta = entry.get("meta") or {}
        self.current_meta_zoom = meta.get("zoom", 15)
        self.graph_original = copy.deepcopy(self.graph)
        self.user_accidents = {}
        self.routes_stale = False
        self.clear_all_accident_entries()
        self.current_routes = []
        self.active_route_index = 0
        default_destination = self.destination_defaults[0] if self.destination_defaults else None
        self.draw_map(origin=self.origin_default, destination=default_destination)
        self.mark_routes_stale("Map changed. Run routing.")

    def draw_map(self, highlighted_paths=None, origin=None, destination=None):
        if self.map_widget is None:
            return
        self.stop_animation()
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()
        self.map_markers = []
        self.route_paths = []

        if not self.coords:
            self.map_widget.set_position(1.557, 110.343)
            self.map_widget.set_zoom(13)
            return

        lats = [lat for (_, lat) in self.coords.values()]
        lons = [lon for (lon, _) in self.coords.values()]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        self.map_widget.set_position(center_lat, center_lon)
        self.map_widget.set_zoom(self.current_meta_zoom or 15)

        for node in sorted(self.coords.keys()):
            lon, lat = self.coords[node]
            label = self.landmarks.get(node, str(node))
            if node == origin:
                display_label = f"Origin: {label}"
            elif node == destination:
                display_label = f"Destination: {label}"
            else:
                display_label = label
            marker = self.map_widget.set_marker(lat, lon, text=display_label)
            self.map_markers.append(marker)

        self.render_base_paths()

        if not highlighted_paths:
            return

        self.route_paths = []

    def render_base_paths(self):
        if self.map_widget is None:
            return
        base_edges = set()
        if self.edge_polylines:
            base_edges.update(self.edge_polylines.keys())
        if self.graph_original:
            for u, neighbors in self.graph_original.items():
                for nbr, _cost in neighbors:
                    base_edges.add((u, nbr))
        for edge in sorted(base_edges):
            u, v = edge
            points = self.build_route_polyline([u, v])
            if len(points) < 2:
                continue
            try:
                self.map_widget.set_path(points, color="#b4b4b4", width=2)
            except Exception:
                continue

    def build_route_polyline(self, node_path):
        if not node_path:
            return []
        latlon_points = []
        for idx in range(len(node_path) - 1):
            u = node_path[idx]
            v = node_path[idx + 1]
            polyline = self.edge_polylines.get((u, v))
            # If the map only stored one direction, try the reverse and flip it.
            if not polyline or len(polyline) < 2:
                reverse = self.edge_polylines.get((v, u))
                if reverse and len(reverse) >= 2:
                    polyline = list(reversed(reverse))
            segment = []
            if polyline:
                segment = [(lat, lon) for lon, lat in polyline]
            else:
                start = self.coords.get(u)
                end = self.coords.get(v)
                if start:
                    segment.append((start[1], start[0]))
                if end:
                    segment.append((end[1], end[0]))
            for point in segment:
                if latlon_points and latlon_points[-1] == point:
                    continue
                latlon_points.append(point)
        if len(latlon_points) < 2:
            for node in node_path:
                coord = self.coords.get(node)
                if coord:
                    point = (coord[1], coord[0])
                    if latlon_points and latlon_points[-1] == point:
                        continue
                    latlon_points.append(point)
        return latlon_points

    def render_active_route(self):
        highlighted = []
        if self.current_routes and 0 <= self.active_route_index < len(self.current_routes):
            highlighted = [self.current_routes[self.active_route_index]["path"]]
        self.draw_map(highlighted, self.last_origin_id, self.last_destination_id)
        self.start_animation()
        self.render_route_cards()

    def stop_animation(self):
        if self.animation_after_id is not None and self.map_widget is not None:
            try:
                self.map_widget.after_cancel(self.animation_after_id)
            except Exception:
                pass
        self.animation_after_id = None
        if self.animation_path_handle is not None:
            try:
                self.animation_path_handle.delete()
            except Exception:
                pass
        self.animation_path_handle = None
        self.animation_points = []
        self.animation_step = 0

    def start_animation(self):
        if not self.current_routes or self.map_widget is None:
            return
        if self.active_route_index < 0 or self.active_route_index >= len(self.current_routes):
            return
        path = self.current_routes[self.active_route_index]["path"]
        points = self.build_route_polyline(path)
        if len(points) < 2:
            return
        self.animation_points = points
        self.animation_step = 1
        self.animation_color = self.route_color(self.active_route_index)
        self.advance_animation()

    def advance_animation(self):
        if not self.animation_points or self.map_widget is None:
            return
        if self.animation_step >= len(self.animation_points):
            self.animation_after_id = None
            return
        partial = self.animation_points[: self.animation_step + 1]
        if self.animation_path_handle is not None:
            try:
                self.animation_path_handle.delete()
            except Exception:
                pass
        self.animation_path_handle = self.map_widget.set_path(partial, color=self.animation_color, width=5)
        self.animation_step += 1
        self.animation_after_id = self.map_widget.after(80, self.advance_animation)

    def discover_map_files(self):
        entries = []
        if not os.path.isdir(self.MAP_DIR):
            return entries
        for filename in sorted(os.listdir(self.MAP_DIR)):
            if not filename.endswith(".txt"):
                continue
            if filename.startswith("heritage_assignment"):
                continue
            path = os.path.join(self.MAP_DIR, filename)
            base = os.path.splitext(filename)[0]
            meta_path = os.path.join(self.MAP_DIR, f"{base}.meta.json")
            image_path = os.path.join(self.MAP_DIR, f"{base}.png")
            paths_path = os.path.join(self.MAP_DIR, f"{base}.paths.json")
            meta = self.load_map_metadata(meta_path)
            if meta and meta.get("image"):
                candidate = os.path.join(self.MAP_DIR, meta["image"])
                if os.path.exists(candidate):
                    image_path = candidate
            entry = {
                "label": self.pretty_map_label(filename),
                "map_path": path,
                "image_path": image_path if os.path.exists(image_path) else None,
                "paths_path": paths_path if os.path.exists(paths_path) else None,
                "meta": meta,
            }
            entries.append(entry)
        return entries

    def pretty_map_label(self, filename):
        base = os.path.splitext(filename)[0]
        label = base.replace("_", " ").title()
        return label.replace("Osm", "OSM")

    def landmark_names(self):
        if not self.landmarks:
            return []
        return [self.landmarks[node] for node in sorted(self.landmarks)]

    def refresh_landmark_menus(self):
        names = self.landmark_names()
        if not names:
            names = ["(none)"]
        for var, menu in [(self.origin_var, self.origin_menu), (self.destination_var, self.destination_menu)]:
            menu["menu"].delete(0, "end")
            for name in names:
                menu["menu"].add_command(label=name, command=lambda value=name, v=var: v.set(value))
        if names[0] == "(none)":
            self.origin_var.set("(none)")
            self.destination_var.set("(none)")
        else:
            if self.origin_var.get() not in names:
                self.origin_var.set("-- choose origin --")
            if self.destination_var.get() not in names:
                self.destination_var.set("-- choose destination --")
        self.refresh_accident_entry_menus()

    def mark_routes_stale(self, reason="Graph changed."):
        self.routes_stale = True
        self.routing_status_var.set(f"Routes stale: {reason} Click 'Run Routing'.")

    def mark_routes_fresh(self):
        self.routes_stale = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.routing_status_var.set(f"Routes up to date (last run {timestamp}).")

    def rebuild_graph_with_accidents(self):
        if not self.graph_original:
            return

        # Start with a clean copy of the original graph
        new_graph = copy.deepcopy(self.graph_original)

        # Apply multipliers to both directions of affected edges
        for (u, v), info in self.user_accidents.items():
            multiplier = info["multiplier"]

            # Forward direction (u -> v)
            if u in new_graph:
                new_graph[u] = [
                    (nbr, cost * multiplier if nbr == v else cost)
                    for nbr, cost in new_graph[u]
                ]

            # Reverse direction (v -> u)
            if v in new_graph:
                new_graph[v] = [
                    (nbr, cost * multiplier if nbr == u else cost)
                    for nbr, cost in new_graph[v]
                ]

        # Store updated graph
        self.graph = new_graph

        # Mark routing stale and refresh UI
        self.mark_routes_stale("Accident slowdowns applied.")
        self.render_active_route()

    def on_map_change(self, selection):
        for entry in self.map_entries:
            if entry["label"] == selection:
                self.map_var.set(selection)
                self.load_map_data(entry)
                self.route_output.delete("1.0", tk.END)
                self.draw_map()
                return

    def load_map_metadata(self, meta_path):
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def load_background_assets(self, entry):
        self.map_background_image = None
        self.background_offset = (0, 0)
        self.background_display_size = (self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        self.current_extent = None
        self.current_projection = None

        image_path = entry.get("image_path")
        paths_path = entry.get("paths_path")
        meta = entry.get("meta")

        if image_path and os.path.exists(image_path):
            try:
                raw = Image.open(image_path)
                scale = min(
                    self.CANVAS_WIDTH / raw.width,
                    self.CANVAS_HEIGHT / raw.height,
                )
                new_size = (int(raw.width * scale), int(raw.height * scale))
                resized = raw.resize(new_size, Image.LANCZOS)
                offset_x = (self.CANVAS_WIDTH - new_size[0]) // 2
                offset_y = (self.CANVAS_HEIGHT - new_size[1]) // 2
                canvas_img = Image.new("RGB", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), (240, 240, 240))
                canvas_img.paste(resized, (offset_x, offset_y))
                self.map_background_image = ImageTk.PhotoImage(canvas_img)
                self.background_offset = (offset_x, offset_y)
                self.background_display_size = new_size
            except Exception:
                self.map_background_image = None
                self.background_offset = (0, 0)
                self.background_display_size = (self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        else:
            self.map_background_image = None
            self.background_offset = (0, 0)
            self.background_display_size = (self.CANVAS_WIDTH, self.CANVAS_HEIGHT)

        if paths_path and os.path.exists(paths_path):
            self.edge_polylines = self.load_edge_polylines(paths_path)
        else:
            self.edge_polylines = {}

        if meta:
            self.current_extent = meta.get("extent")
            self.current_projection = meta.get("projection")

    def load_edge_polylines(self, path_file):
        try:
            with open(path_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return {}

        polylines = {}
        for key, points in data.items():
            try:
                u_str, v_str = key.split("-")
                u = int(u_str)
                v = int(v_str)
            except ValueError:
                continue
            polylines[(u, v)] = [(pt["lon"], pt["lat"]) for pt in points]
        # Also synthesize reverse-direction polylines when missing, to avoid
        # straight-line fallback for graphs that contain both directions but
        # the geometry was stored only once.
        for (u, v), pts in list(polylines.items()):
            if (v, u) not in polylines and pts and len(pts) >= 2:
                polylines[(v, u)] = list(reversed(pts))
        return polylines

    @staticmethod
    def lonlat_to_webmerc(lon, lat):
        lat = max(min(lat, 89.9), -89.9)
        r = 6378137.0
        x = r * math.radians(lon)
        y = r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return x, y

    def route_color(self, idx):
        palette = ["#ff8c42", "#3a86ff", "#8ac926", "#ff006e", "#8338ec"]
        return palette[idx % len(palette)]

    @staticmethod
    def format_travel_time(hours):
        minutes = hours * 60.0
        if minutes >= 100:
            return f"{minutes:.0f} min"
        if minutes >= 10:
            return f"{minutes:.1f} min"
        return f"{minutes:.2f} min"

    def render_route_cards(self):
        if self.route_cards_container is None:
            return
        for child in self.route_cards_container.winfo_children():
            child.destroy()
        if not self.current_routes:
            tk.Label(
                self.route_cards_container,
                text="Run routing to view route options.",
                bg="#f5f5f5",
                fg="#666",
                anchor="w",
                wraplength=320,
                justify="left",
            ).pack(fill="x")
            return

        for idx, info in enumerate(self.current_routes, start=1):
            is_active = (idx - 1) == self.active_route_index
            bg = "#e8f0ff" if is_active else "#ffffff"
            border = "#4c8bf5" if is_active else "#cccccc"
            card = tk.Frame(
                self.route_cards_container,
                bg=bg,
                highlightbackground=border,
                highlightthickness=1,
                padx=8,
                pady=6,
            )
            card.pack(fill="x", pady=4)

            title = tk.Label(
                card,
                text=f"Route {idx} · {self.format_travel_time(info['cost'])}",
                font=("Arial", 11, "bold"),
                bg=bg,
                anchor="w",
            )
            title.pack(fill="x")

            path_nodes = " → ".join(self.node_label(n) for n in info["path"])
            details = tk.Label(
                card,
                text=path_nodes,
                bg=bg,
                fg="#555",
                wraplength=320,
                justify="left",
                anchor="w",
            )
            details.pack(fill="x", pady=(2, 0))

            card.bind("<Button-1>", lambda _e, i=idx - 1: self.on_route_card_clicked(i))
            title.bind("<Button-1>", lambda _e, i=idx - 1: self.on_route_card_clicked(i))
            details.bind("<Button-1>", lambda _e, i=idx - 1: self.on_route_card_clicked(i))

    def on_route_card_clicked(self, idx):
        if idx == self.active_route_index or idx < 0 or idx >= len(self.current_routes):
            return
        self.active_route_index = idx
        self.render_active_route()
        self.render_route_cards()

    def resolve_node(self, name):
        node_id = self.name_to_id.get(name)
        if node_id is None:
            raise ValueError(f"Unknown landmark: {name}")
        return node_id

    def node_label(self, node):
        label = self.landmarks.get(node)
        return f"{node} ({label})" if label else str(node)

    def run_routing(self):
        if not self.graph:
            messagebox.showwarning("Missing Map", "Map data is not loaded.")
            return
        self.sync_accidents_to_graph()

        origin_name = self.origin_var.get()
        destination_name = self.destination_var.get()

        if origin_name.startswith("--"):
            messagebox.showwarning("Origin missing", "Please choose an origin landmark.")
            return
        if destination_name.startswith("--"):
            messagebox.showwarning("Destination missing", "Please choose a destination landmark.")
            return

        try:
            origin_id = self.resolve_node(origin_name)
            destination_id = self.resolve_node(destination_name)
        except ValueError as exc:
            messagebox.showerror("Invalid selection", str(exc))
            return

        try:
            k = max(1, min(5, int(self.k_var.get())))
        except ValueError:
            k = 1

        selected_label = self.algorithm_var.get()
        method = self.algorithm_key_lookup.get(selected_label, "CUS1").upper()
        if method != "CUS1" and k > 1:
            k = 1

        self.current_routes = []
        self.active_route_index = 0
        self.last_origin_id = origin_id
        self.last_destination_id = destination_id

        self.route_output.delete("1.0", tk.END)

        if method == "CUS1" and k > 1:
            routes, nodes_expanded = yen_k_shortest_paths(self.graph, origin_id, destination_id, k)
            if not routes:
                self.route_output.insert(
                    tk.END,
                    f"No path found from {origin_name} to {destination_name}.\n"
                )
                self.draw_map(origin=origin_id, destination=destination_id)
                self.render_route_cards()
                return
            self.current_routes = routes
        else:
            try:
                path, cost, nodes_expanded = self.run_single_search(
                    method,
                    origin_id,
                    [destination_id],
                )
            except ValueError as exc:
                messagebox.showerror("Error", str(exc))
                return
            if not path:
                self.route_output.insert(
                    tk.END,
                    f"No path found from {origin_name} to {destination_name}.\n"
                )
                self.draw_map(origin=origin_id, destination=destination_id)
                self.render_route_cards()
                return
            self.current_routes = [{"path": path, "cost": cost}]
            self.active_route_index = 0

        self.render_active_route()

        lines = [
            f"Origin: {self.node_label(origin_id)}",
            f"Destination: {self.node_label(destination_id)}",
        ]

        if self.user_accidents:
            applied = ", ".join(
                f"{self.node_label(u)} → {self.node_label(v)} ({info['severity']}, ×{info['multiplier']:.2f})"
                for (u, v), info in self.user_accidents.items()
            )
            lines.append(f"Accidents detected: {applied}")
        else:
            lines.append("Accidents detected: None")

        lines += [
            "",
            "Routes:",
        ]

        for idx, info in enumerate(self.current_routes, start=1):
            path_nodes = " -> ".join(self.node_label(n) for n in info["path"])
            lines.append(f"{idx}) {path_nodes}")
            lines.append(f"    Travel time: {self.format_travel_time(info['cost'])}")

        lines.append("")
        lines.append(f"Nodes expanded: {nodes_expanded}")
        if self.user_accidents:
            lines.append("")
            lines.append("Applied accident slowdowns:")
            for (u, v), info in sorted(self.user_accidents.items()):
                lines.append(
                    f"- {self.node_label(u)} -> {self.node_label(v)}: {info['severity']} × {info['multiplier']:.2f}"
                )
                formula_text = info.get("formula")
                if formula_text:
                    lines.append(f"{formula_text}")

        self.route_output.insert(tk.END, "\n".join(lines))
        self.mark_routes_fresh()

    def run_single_search(self, method, origin_id, destinations):
        method = method.upper()
        if method == "CUS1":
            return cus1_ucs(self.graph, origin_id, destinations)
        if method == "DFS":
            return dfs_search(self.graph, origin_id, destinations)
        if method == "BFS":
            return bfs_search(self.graph, origin_id, destinations)
        if method == "GBFS":
            result = gbfs_search(self.graph, origin_id, destinations, self.coords)

            # If already (path, cost, nodes)
            if len(result) == 3:
                return result

            # Otherwise unpack and compute cost manually
            path, nodes = result
            cost = self.compute_path_cost(path)
            return path, cost, nodes

        if method == "ASTAR":
            result = astar_search(self.graph, origin_id, destinations, self.coords)

            if len(result) == 3:  
                # (path, cost, nodes)
                return result

            # Otherwise compute cost
            path, nodes = result
            cost = self.compute_path_cost(path)
            return path, cost, nodes

        if method == "CUS2":
            result = cus2_hcs(self.graph, origin_id, destinations, self.coords)

            if len(result) == 3:
                return result

            path, nodes = result
            cost = self.compute_path_cost(path)
            return path, cost, nodes

        raise ValueError(f"Unknown algorithm '{method}'")

    def compute_path_cost(self, path):
        if not path or len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for nbr, cost in self.graph[u]:
                if nbr == v:
                    total += cost
                    break
        return total

def main():
    root = tk.Tk()
    ICS_GUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()


if __name__ == "__main__":
    main()
