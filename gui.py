import tkinter as tk
from tkinter import filedialog, messagebox

import copy
import json
import math
import os

from PIL import Image, ImageTk

from graph import load_graph
from topk import yen_k_shortest_paths
from algorithms.cus1_ucs import cus1_ucs
from algorithms.dfs import dfs_search
from algorithms.bfs import bfs_search
from algorithms.gbfs import gbfs_search
from algorithms.astar import astar_search
from algorithms.cus2_hcs import cus2_hcs

# =============================
# ICS GUI (Assignment 2B)
# =============================
class ICS_GUI:
    SEVERITY_LEVELS = {
        "Minor": 1.15,
        "Moderate": 1.35,
        "Severe": 1.75,
    }
    CANVAS_WIDTH = 760
    CANVAS_HEIGHT = 700
    MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")

    def __init__(self, root):
        self.root = root
        self.root.title("ICS – Incident Classification System")
        self.root.geometry("1350x750")

        # ML prediction placeholders
        self.uploaded_img = None
        self.cnn_model = None
        self.model2 = None
        self.current_severity = None

        self.graph = {}
        self.graph_original = {}
        self.origin_default = None
        self.destination_defaults = []
        self.coords = {}
        self.accident = {}
        self.user_accidents = {}
        self.landmarks = {}
        self.name_to_id = {}
        self.canvas_points = {}
        self.current_paths = []
        self.accident_origin_menu = None
        self.accident_target_menu = None
        self.accident_list_keys = []
        self.routes_stale = False
        self.cached_accident_edge_labels = ["(no edges available)"]

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
        self.animation_in_progress = False
        self.animation_points = []
        self.animation_step = 0
        self.animation_line_ids = []
        self.animation_after_id = None

        self.current_map_entry = self.map_entries[0]
        self.map_var = tk.StringVar(value=self.current_map_entry["label"])
        self.map_label_var = tk.StringVar()
        self.accident_origin_var = tk.StringVar(value="(origin)")
        self.accident_target_var = tk.StringVar(value="(neighbor)")
        self.accident_status_var = tk.StringVar(value="")
        self.accident_listbox = None
        self.accident_origin_menu = None
        self.accident_target_menu = None
        self.current_accident_origin = None
        self.current_accident_target = None

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

        self.map_canvas = tk.Canvas(
            left,
            bg="white",
            highlightbackground="#ccc",
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
        )
        self.map_canvas.pack(fill="both", expand=True)

        # ============================
        # RIGHT PANEL (SCROLLABLE)
        # ============================

        # Outer frame holding canvas + scrollbar
        right_outer = tk.Frame(self.root, width=480, bg="#f5f5f5")
        right_outer.pack(side="right", fill="y")

        # Canvas used for scrolling
        right_canvas = tk.Canvas(right_outer, bg="#f5f5f5", width=480)
        right_canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        right_canvas.configure(yscrollcommand=scrollbar.set)
        right_canvas.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )

        # Actual content frame placed INSIDE the canvas
        right = tk.Frame(right_canvas, bg="#f5f5f5", padx=20, pady=20)
        right_canvas.create_window((0, 0), window=right, anchor="nw")

        # ==============================
        # RIGHT SIDE CONTENT (unchanged)
        # ==============================
        tk.Label(right, text="Accident Image Classification",
                font=("Arial", 16, "bold")).pack(pady=10)

        self.preview = tk.Label(right, text="No Image Uploaded",
                                bg="#ddd", width=40, height=12)
        self.preview.pack(pady=10)

        tk.Button(right, text="Upload Image",
                command=self.upload_image,
                width=25).pack(pady=5)

        self.cnn_label = tk.Label(right, text="CNN Prediction: -",
                                font=("Arial", 12), anchor="w")
        self.cnn_label.pack(fill="x", pady=5)

        self.model2_label = tk.Label(right, text="Model 2 Prediction: -",
                                    font=("Arial", 12), anchor="w")
        self.model2_label.pack(fill="x", pady=5)

        self.final_label = tk.Label(right, text="Final Severity: -",
                                    font=("Arial", 13, "bold"),
                                    fg="darkred")
        self.final_label.pack(fill="x", pady=10)

        tk.Button(right, text="Run Models",
                command=self.run_ml_prediction,
                width=25).pack(pady=5)

        tk.Label(right, text="---------------------------------").pack(pady=10)

        tk.Label(right, text="Route Finder", font=("Arial", 16, "bold")).pack()

        selections = tk.Frame(right, bg="#f5f5f5")
        selections.pack(fill="x", pady=5)

        tk.Label(selections, text="Origin:", anchor="w",
                bg="#f5f5f5").grid(row=0, column=0, sticky="w")
        origin_names = self.landmark_names() or ["(none)"]
        self.origin_var = tk.StringVar(value=origin_names[0])
        self.origin_menu = tk.OptionMenu(selections, self.origin_var, *origin_names)
        self.origin_menu.grid(row=0, column=1, sticky="ew")

        tk.Label(selections, text="Destination:", anchor="w",
                bg="#f5f5f5").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.destination_var = tk.StringVar(value=origin_names[-1])
        self.destination_menu = tk.OptionMenu(selections, self.destination_var, *origin_names)
        self.destination_menu.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        tk.Label(selections, text="Number of routes (k):", anchor="w",
                bg="#f5f5f5").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.k_var = tk.StringVar(value="3")
        tk.Spinbox(selections, from_=1, to=5, textvariable=self.k_var, width=5).grid(row=2, column=1, sticky="w")
        tk.Label(selections, text="Display route:", anchor="w",
                bg="#f5f5f5").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.route_choice_var = tk.StringVar(value="Route 1")
        self.route_choice_menu = tk.OptionMenu(
            selections,
            self.route_choice_var,
            "Route 1",
            command=self.on_route_option_selected,
        )
        self.route_choice_menu.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(4, 0))
        self.route_choice_menu.config(state="disabled")

        tk.Label(selections, text="Map:", anchor="w", bg="#f5f5f5").grid(row=4, column=0, sticky="w", pady=(8, 0))
        map_labels = [entry["label"] for entry in self.map_entries]
        self.map_menu = tk.OptionMenu(selections, self.map_var, *map_labels, command=self.on_map_change)
        self.map_menu.grid(row=4, column=1, sticky="ew", pady=(8, 0))

        tk.Label(selections, text="Algorithm:", anchor="w", bg="#f5f5f5").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.algorithm_var = tk.StringVar(value="CUS1")
        algorithms = ["CUS1", "DFS", "BFS", "GBFS", "ASTAR", "CUS2"]
        self.algorithm_menu = tk.OptionMenu(selections, self.algorithm_var, *algorithms)
        self.algorithm_menu.grid(row=5, column=1, sticky="ew", pady=(8, 0))

        selections.grid_columnconfigure(1, weight=1)

        accident_frame = tk.LabelFrame(
            right,
            text="Accident Overrides",
            bg="#f5f5f5",
            padx=10,
            pady=10,
        )
        accident_frame.pack(fill="x", pady=10)

        tk.Label(
            accident_frame,
            text="Accident Origin:",
            bg="#f5f5f5",
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.accident_origin_menu = tk.OptionMenu(accident_frame, self.accident_origin_var, "(no nodes)")
        self.accident_origin_menu.grid(row=0, column=1, sticky="ew")

        tk.Label(
            accident_frame,
            text="Target:",
            bg="#f5f5f5",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.accident_target_menu = tk.OptionMenu(accident_frame, self.accident_target_var, "(no neighbors)")
        self.accident_target_menu.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        tk.Button(
            accident_frame,
            text="Add / Update",
            command=self.add_custom_accident,
            width=20,
        ).grid(row=2, column=0, columnspan=2, pady=(8, 4))

        list_frame = tk.Frame(accident_frame, bg="#f5f5f5")
        list_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(4, 4))
        accident_frame.grid_columnconfigure(1, weight=1)
        accident_frame.grid_rowconfigure(3, weight=1)

        list_scroll = tk.Scrollbar(list_frame, orient="vertical")
        self.accident_listbox = tk.Listbox(list_frame, height=5, yscrollcommand=list_scroll.set)
        self.accident_listbox.pack(side="left", fill="both", expand=True)
        list_scroll.config(command=self.accident_listbox.yview)
        list_scroll.pack(side="right", fill="y")

        btn_row = tk.Frame(accident_frame, bg="#f5f5f5")
        btn_row.grid(row=4, column=0, columnspan=2, pady=(4, 0))

        tk.Button(
            btn_row,
            text="Remove Selected",
            command=self.remove_selected_accident,
            width=16,
        ).pack(side="left", padx=(0, 6))

        tk.Button(
            btn_row,
            text="Clear All",
            command=self.clear_custom_accidents,
            width=10,
        ).pack(side="left")

        self.accident_status_var.set("")
        self.recompute_accident_edges()

        tk.Button(right, text="Run Routing",
                command=self.run_routing,
                width=25).pack(pady=10)

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
    # Image Upload
    # ---------------------------
    def upload_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not filename:
            return

        img = Image.open(filename).resize((300, 300))
        self.uploaded_img = img
        self.preview_imgtk = ImageTk.PhotoImage(img)
        self.preview.config(image=self.preview_imgtk, text="")

    # ---------------------------
    # ML Prediction Pipeline
    # ---------------------------
    def run_ml_prediction(self):
        if self.uploaded_img is None:
            messagebox.showwarning("Error", "Upload an accident image first.")
            return

        # TODO: Your Team Will Implement These:
        cnn_result = "Minor"          # predict_cnn(self.uploaded_img)
        model2_result = "Moderate"    # predict_model2(self.uploaded_img)
        final_result = "Moderate"     # combine(cnn_result, model2_result)

        # Update UI
        self.cnn_label.config(text=f"CNN Prediction: {cnn_result}")
        self.model2_label.config(text=f"Model 2 Prediction: {model2_result}")
        self.final_label.config(text=f"Final Severity: {final_result}")

        # Store for routing
        self.current_severity = final_result

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
        self.compute_canvas_points()
        if hasattr(self, "origin_menu"):
            self.refresh_landmark_menus()
        self.graph_original = copy.deepcopy(self.graph)
        self.user_accidents = {}
        self.routes_stale = False
        self.accident_status_var.set("No custom accidents.")
        self.recompute_accident_edges()
        self.refresh_accident_listbox()
        self.current_routes = []
        self.active_route_index = 0
        if hasattr(self, "route_selector_frame"):
            self.update_route_selector()

    def compute_canvas_points(self):
        if not self.coords:
            self.canvas_points = {}
            return

        using_projection = bool(self.current_extent and self.current_projection == "web_mercator")
        if using_projection:
            min_x = self.current_extent["xmin"]
            max_x = self.current_extent["xmax"]
            min_y = self.current_extent["ymin"]
            max_y = self.current_extent["ymax"]
            display_width, display_height = self.background_display_size
            offset_x, offset_y = self.background_offset
            span_x = max(max_x - min_x, 1e-5)
            span_y = max(max_y - min_y, 1e-5)
            scale_x = display_width / span_x
            scale_y = display_height / span_y

            self.canvas_points = {}
            for node, (lon, lat) in self.coords.items():
                x_m, y_m = self.lonlat_to_webmerc(lon, lat)
                px = offset_x + (x_m - min_x) * scale_x
                py = offset_y + display_height - (y_m - min_y) * scale_y
                self.canvas_points[node] = (px, py)
            return

        xs = [pt[0] for pt in self.coords.values()]
        ys = [pt[1] for pt in self.coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = max(max_x - min_x, 1e-5)
        span_y = max(max_y - min_y, 1e-5)
        margin = 40
        scale = min(
            (self.CANVAS_WIDTH - 2 * margin) / span_x,
            (self.CANVAS_HEIGHT - 2 * margin) / span_y,
        )
        offset_x = (self.CANVAS_WIDTH - span_x * scale) / 2
        offset_y = (self.CANVAS_HEIGHT - span_y * scale) / 2

        self.canvas_points = {}
        for node, (lon, lat) in self.coords.items():
            px = offset_x + (lon - min_x) * scale
            py = self.CANVAS_HEIGHT - (offset_y + (lat - min_y) * scale)
            self.canvas_points[node] = (px, py)

    def draw_map(self, highlighted_paths=None, origin=None, destination=None):
        self.map_canvas.delete("all")
        if self.animation_in_progress:
            self.animation_in_progress = False
            self.animation_line_ids = []
            if self.animation_after_id is not None:
                self.map_canvas.after_cancel(self.animation_after_id)
                self.animation_after_id = None
        if self.map_background_image:
            self.map_canvas.create_image(
                0,
                0,
                image=self.map_background_image,
                anchor="nw",
            )

        if not self.canvas_points:
            self.map_canvas.create_text(
                self.CANVAS_WIDTH / 2,
                self.CANVAS_HEIGHT / 2,
                text="No map data loaded",
                fill="gray",
            )
            return

        highlighted_edges = {}
        if highlighted_paths:
            for idx, path in enumerate(highlighted_paths):
                for i in range(len(path) - 1):
                    highlighted_edges[(path[i], path[i + 1])] = idx

        # Pass 1: draw base edges
        for u, neighbors in self.graph.items():
            for v, _ in neighbors:
                polyline = self.edge_polylines.get((u, v))
                if polyline:
                    self.draw_polyline(polyline, is_highlighted=False)
                else:
                    x1, y1 = self.canvas_points.get(u, (0, 0))
                    x2, y2 = self.canvas_points.get(v, (0, 0))
                    self.map_canvas.create_line(x1, y1, x2, y2, fill="#d0d0d0", width=2)

        # Pass 2: draw highlighted edges on top
        for edge, idx in highlighted_edges.items():
            u, v = edge
            polyline = self.edge_polylines.get(edge)
            if polyline:
                self.draw_polyline(
                    polyline,
                    is_highlighted=True,
                    route_index=idx,
                )
            else:
                x1, y1 = self.canvas_points.get(u, (0, 0))
                x2, y2 = self.canvas_points.get(v, (0, 0))
                color = self.route_color(idx)
                self.map_canvas.create_line(x1, y1, x2, y2, fill=color, width=4)

        for node, (x, y) in self.canvas_points.items():
            fill = "#0077b6"
            radius = 6
            if node == origin:
                fill = "#2a9d8f"
                radius = 7
            elif node == destination:
                fill = "#e63946"
                radius = 7
            self.map_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline="white",
                width=2,
            )
            label = self.landmarks.get(node, str(node))
            self.map_canvas.create_text(
                x + 10,
                y - 10,
                text=label,
                anchor="w",
                font=("Arial", 9),
                fill="#333",
            )

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
                self.origin_var.set(names[0])
            if self.destination_var.get() not in names:
                fallback = names[1] if len(names) > 1 else names[0]
                self.destination_var.set(fallback)

    def recompute_accident_edges(self):
        self.update_accident_origin_menu()

    def update_accident_origin_menu(self):
        if self.accident_origin_menu is None:
            return
        menu = self.accident_origin_menu["menu"]
        menu.delete(0, "end")

        if not self.graph_original:
            self.accident_origin_var.set("(no nodes)")
            menu.add_command(label="(no nodes)", command=lambda: None)
            self.current_accident_origin = None
            self.update_accident_target_menu(None)
            return

        nodes = sorted(self.graph_original.keys())
        if not nodes:
            self.accident_origin_var.set("(no nodes)")
            menu.add_command(label="(no nodes)", command=lambda: None)
            self.current_accident_origin = None
            self.update_accident_target_menu(None)
            return

        for node in nodes:
            label = self.node_label(node)
            menu.add_command(
                label=label,
                command=lambda value=node: self.on_accident_origin_selected(value),
            )
        self.on_accident_origin_selected(nodes[0])

    def on_accident_origin_selected(self, node_id):
        self.current_accident_origin = node_id
        self.accident_origin_var.set(self.node_label(node_id))
        neighbors = self.graph_original.get(node_id, [])
        self.update_accident_target_menu(neighbors)

    def update_accident_target_menu(self, neighbors):
        if self.accident_target_menu is None:
            return
        menu = self.accident_target_menu["menu"]
        menu.delete(0, "end")

        if not neighbors:
            self.accident_target_var.set("(no neighbors)")
            menu.add_command(label="(no neighbors)", command=lambda: None)
            self.current_accident_target = None
            return

        for nbr, cost in sorted(neighbors, key=lambda item: item[0]):
            label = f"{self.node_label(nbr)} ({cost:.3f} h)"
            menu.add_command(
                label=label,
                command=lambda value=nbr: self.on_accident_target_selected(value),
            )
        self.on_accident_target_selected(neighbors[0][0])

    def on_accident_target_selected(self, node_id):
        self.current_accident_target = node_id
        self.accident_target_var.set(self.node_label(node_id))

    def refresh_accident_listbox(self):
        if self.accident_listbox is None:
            return
        self.accident_listbox.delete(0, tk.END)
        self.accident_list_keys = []
        for edge in sorted(self.user_accidents.keys()):
            info = self.user_accidents[edge]
            u, v = edge
            label = f"{self.node_label(u)} -> {self.node_label(v)} ({info['severity']} ×{info['multiplier']:.2f})"
            self.accident_listbox.insert(tk.END, label)
            self.accident_list_keys.append(edge)

    def add_custom_accident(self):
        if not self.graph_original:
            messagebox.showwarning("No graph", "Load a map before adding accidents.")
            return
        if self.current_accident_origin is None or self.current_accident_target is None:
            messagebox.showwarning("Invalid edge", "Select an origin and target node.")
            return
        edge = (self.current_accident_origin, self.current_accident_target)
        multiplier = self.SEVERITY_LEVELS.get("Moderate", 1.35)
        self.user_accidents[edge] = {"severity": "Auto", "multiplier": multiplier}
        self.refresh_accident_listbox()
        self.rebuild_graph_with_accidents()

    def remove_selected_accident(self):
        if not self.accident_list_keys:
            return
        selection = self.accident_listbox.curselection()
        if not selection:
            messagebox.showinfo("Select entry", "Choose an accident entry to remove.")
            return
        idx = selection[0]
        edge = self.accident_list_keys[idx]
        if edge in self.user_accidents:
            del self.user_accidents[edge]
        self.refresh_accident_listbox()
        self.rebuild_graph_with_accidents()

    def clear_custom_accidents(self):
        if not self.user_accidents:
            return
        self.user_accidents.clear()
        self.refresh_accident_listbox()
        self.rebuild_graph_with_accidents()

    def rebuild_graph_with_accidents(self):
        if not self.graph_original:
            return
        new_graph = {
            node: [(nbr, cost) for nbr, cost in neighbors]
            for node, neighbors in self.graph_original.items()
        }
        for (u, v), info in self.user_accidents.items():
            if u not in new_graph:
                continue
            updated = []
            applied = False
            for nbr, cost in new_graph[u]:
                if nbr == v:
                    updated.append((nbr, cost * info["multiplier"]))
                    applied = True
                else:
                    updated.append((nbr, cost))
            if applied:
                new_graph[u] = updated
        self.graph = new_graph
        self.routes_stale = True

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
        return polylines

    @staticmethod
    def lonlat_to_webmerc(lon, lat):
        lat = max(min(lat, 89.9), -89.9)
        r = 6378137.0
        x = r * math.radians(lon)
        y = r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return x, y

    def polyline_to_canvas(self, polyline):
        using_projection = bool(self.current_extent and self.current_projection == "web_mercator")
        points = []
        if using_projection:
            min_x = self.current_extent["xmin"]
            max_x = self.current_extent["xmax"]
            min_y = self.current_extent["ymin"]
            max_y = self.current_extent["ymax"]
            display_width, display_height = self.background_display_size
            offset_x, offset_y = self.background_offset
            span_x = max(max_x - min_x, 1e-5)
            span_y = max(max_y - min_y, 1e-5)
            scale_x = display_width / span_x
            scale_y = display_height / span_y

            for lon, lat in polyline:
                x_m, y_m = self.lonlat_to_webmerc(lon, lat)
                px = offset_x + (x_m - min_x) * scale_x
                py = offset_y + display_height - (y_m - min_y) * scale_y
                points.append((px, py))
        else:
            xs = [pt[0] for pt in self.coords.values()]
            ys = [pt[1] for pt in self.coords.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max(max_x - min_x, 1e-5)
            span_y = max(max_y - min_y, 1e-5)
            margin = 40
            scale = min(
                (self.CANVAS_WIDTH - 2 * margin) / span_x,
                (self.CANVAS_HEIGHT - 2 * margin) / span_y,
            )
            offset_x = (self.CANVAS_WIDTH - span_x * scale) / 2
            offset_y = (self.CANVAS_HEIGHT - span_y * scale) / 2

            for lon, lat in polyline:
                px = offset_x + (lon - min_x) * scale
                py = self.CANVAS_HEIGHT - (offset_y + (lat - min_y) * scale)
                points.append((px, py))

        return points

    def draw_polyline(self, polyline, is_highlighted=False, route_index=0):
        canvas_points = self.polyline_to_canvas(polyline)
        if len(canvas_points) < 2:
            return
        color = self.route_color(route_index) if is_highlighted else "#d0d0d0"
        width = 4 if is_highlighted else 2
        for i in range(len(canvas_points) - 1):
            x1, y1 = canvas_points[i]
            x2, y2 = canvas_points[i + 1]
            self.map_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def route_color(self, idx):
        palette = ["#ff8c42", "#3a86ff", "#8ac926", "#ff006e", "#8338ec"]
        return palette[idx % len(palette)]

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

        origin_name = self.origin_var.get()
        destination_name = self.destination_var.get()

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

        method = self.algorithm_var.get().upper()
        if method != "CUS1" and k > 1:
            messagebox.showinfo(
                "Top-k not available",
                "Multiple routes are currently supported only for CUS1. "
                "Proceeding with k=1.",
            )
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
                self.update_route_selector()
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
                self.update_route_selector()
                return
            self.current_routes = [{"path": path, "cost": cost}]
            self.active_route_index = 0

        self.draw_map(None, origin_id, destination_id)
        self.update_route_selector()
        self.start_animation()

        lines = [
            f"Origin: {self.node_label(origin_id)}",
            f"Destination: {self.node_label(destination_id)}",
            f"Accident: {self.accident.get('edge')} ({self.accident.get('severity')})",
            "",
            "Routes:",
        ]

        for idx, info in enumerate(self.current_routes, start=1):
            path_nodes = " -> ".join(self.node_label(n) for n in info["path"])
            lines.append(f"{idx}) {path_nodes}")
            lines.append(f"    Travel time: {info['cost']:.4f}")

        lines.append("")
        lines.append(f"Nodes expanded: {nodes_expanded}")

        self.route_output.insert(tk.END, "\n".join(lines))
        self.routes_stale = False

    def update_route_selector(self):
        if len(self.current_routes) <= 1:
            self.route_choice_var.set("Route 1")
            self.route_choice_menu.config(state="disabled")
            return

        menu = self.route_choice_menu["menu"]
        menu.delete(0, "end")
        for idx, info in enumerate(self.current_routes):
            menu.add_command(
                label=f"Route {idx + 1} ({info['cost']:.4f} h)",
                command=lambda value=idx: self.on_route_option_selected(value),
            )
        self.route_choice_menu.config(state="normal")
        self.route_choice_var.set(f"Route {self.active_route_index + 1} ({self.current_routes[self.active_route_index]['cost']:.4f} h)")

    def on_route_option_selected(self, value):
        idx = 0
        if isinstance(value, str):
            try:
                idx = int(value.split()[1]) - 1
            except Exception:
                idx = 0
        else:
            idx = int(value)
        if idx < 0 or idx >= len(self.current_routes):
            return
        self.active_route_index = idx
        if self.last_origin_id is None or self.last_destination_id is None:
            return
        self.draw_map(None, self.last_origin_id, self.last_destination_id)
        if self.current_routes:
            self.route_choice_var.set(f"Route {idx + 1} ({self.current_routes[idx]['cost']:.4f} h)")
        self.start_animation()

    def run_single_search(self, method, origin_id, destinations):
        method = method.upper()
        if method == "CUS1":
            return cus1_ucs(self.graph, origin_id, destinations)
        if method == "DFS":
            return dfs_search(self.graph, origin_id, destinations)
        if method == "BFS":
            return bfs_search(self.graph, origin_id, destinations)
        if method == "GBFS":
            return gbfs_search(self.graph, origin_id, destinations, self.coords)
        if method == "ASTAR":
            return astar_search(self.graph, origin_id, destinations, self.coords)
        if method == "CUS2":
            return cus2_hcs(self.graph, origin_id, destinations, self.coords)
        raise ValueError(f"Unknown algorithm '{method}'")
    def start_animation(self):
        if not self.current_routes:
            return
        current = self.current_routes[self.active_route_index]["path"]
        if len(current) < 2:
            return
        points = []
        for i in range(len(current) - 1):
            edge = (current[i], current[i + 1])
            polyline = self.edge_polylines.get(edge)
            if not polyline:
                start = self.coords.get(current[i])
                end = self.coords.get(current[i + 1])
                if start and end:
                    polyline = [start, end]
                else:
                    continue
            canvas_points = self.polyline_to_canvas(polyline)
            if not canvas_points:
                continue
            if points:
                if points[-1] != canvas_points[0]:
                    points.extend(canvas_points)
                else:
                    points.extend(canvas_points[1:])
            else:
                points.extend(canvas_points)
        if len(points) < 2:
            return
        if self.animation_after_id is not None:
            self.map_canvas.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        self.animation_points = points
        self.animation_step = 0
        self.animation_line_ids = []
        self.animation_in_progress = True
        self.animation_after_id = self.map_canvas.after(60, self.advance_animation)

    def advance_animation(self):
        if not self.animation_in_progress:
            return
        if self.animation_step >= len(self.animation_points) - 1:
            self.animation_in_progress = False
            self.animation_after_id = None
            return
        x1, y1 = self.animation_points[self.animation_step]
        x2, y2 = self.animation_points[self.animation_step + 1]
        color = self.route_color(self.active_route_index)
        line_id = self.map_canvas.create_line(x1, y1, x2, y2, fill=color, width=4)
        self.animation_line_ids.append(line_id)
        self.animation_step += 1
        self.animation_after_id = self.map_canvas.after(60, self.advance_animation)


def main():
    root = tk.Tk()
    ICS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
