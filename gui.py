import tkinter as tk
from tkinter import filedialog, messagebox

import json
import math
import os

from PIL import Image, ImageTk

from graph import load_graph
from topk import yen_k_shortest_paths

# =============================
# ICS GUI (Assignment 2B)
# =============================
class ICS_GUI:
    CANVAS_WIDTH = 820
    CANVAS_HEIGHT = 720
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
        self.origin_default = None
        self.destination_defaults = []
        self.coords = {}
        self.accident = {}
        self.landmarks = {}
        self.name_to_id = {}
        self.canvas_points = {}
        self.current_paths = []

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

        self.current_map_entry = self.map_entries[0]
        self.map_var = tk.StringVar(value=self.current_map_entry["label"])
        self.map_label_var = tk.StringVar()

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
        self.root.geometry("1250x700")   # slightly smaller, fits screens better

        # ============================
        # LEFT PANEL (Map)
        # ============================
        left = tk.Frame(self.root, width=850, height=700, bg="white")
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
        right_outer = tk.Frame(self.root, width=400, bg="#f5f5f5")
        right_outer.pack(side="right", fill="y")

        # Canvas used for scrolling
        right_canvas = tk.Canvas(right_outer, bg="#f5f5f5", width=400)
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

        tk.Label(selections, text="Map:", anchor="w", bg="#f5f5f5").grid(row=3, column=0, sticky="w", pady=(8, 0))
        map_labels = [entry["label"] for entry in self.map_entries]
        self.map_menu = tk.OptionMenu(selections, self.map_var, *map_labels, command=self.on_map_change)
        self.map_menu.grid(row=3, column=1, sticky="ew", pady=(8, 0))

        selections.grid_columnconfigure(1, weight=1)

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

        highlighted_edges = set()
        if highlighted_paths:
            for path in highlighted_paths:
                for i in range(len(path) - 1):
                    highlighted_edges.add((path[i], path[i + 1]))

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
        for edge in highlighted_edges:
            u, v = edge
            polyline = self.edge_polylines.get(edge)
            if polyline:
                self.draw_polyline(polyline, is_highlighted=True)
            else:
                x1, y1 = self.canvas_points.get(u, (0, 0))
                x2, y2 = self.canvas_points.get(v, (0, 0))
                self.map_canvas.create_line(x1, y1, x2, y2, fill="#ff8c42", width=4)

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

    def draw_polyline(self, polyline, is_highlighted=False):
        canvas_points = self.polyline_to_canvas(polyline)
        if len(canvas_points) < 2:
            return
        color = "#ff8c42" if is_highlighted else "#d0d0d0"
        width = 4 if is_highlighted else 2
        for i in range(len(canvas_points) - 1):
            x1, y1 = canvas_points[i]
            x2, y2 = canvas_points[i + 1]
            self.map_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

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

        paths, nodes_expanded = yen_k_shortest_paths(self.graph, origin_id, destination_id, k)
        self.route_output.delete("1.0", tk.END)

        if not paths:
            self.route_output.insert(
                tk.END,
                f"No path found from {origin_name} to {destination_name}.\n"
            )
            self.draw_map(origin=origin_id, destination=destination_id)
            return

        highlighted_paths = [entry["path"] for entry in paths]
        self.draw_map(highlighted_paths, origin_id, destination_id)

        lines = [
            f"Origin: {self.node_label(origin_id)}",
            f"Destination: {self.node_label(destination_id)}",
            f"Accident: {self.accident.get('edge')} ({self.accident.get('severity')})",
            "",
            "Routes:",
        ]

        for idx, info in enumerate(paths, start=1):
            path_nodes = " -> ".join(self.node_label(n) for n in info["path"])
            lines.append(f"{idx}) {path_nodes}")
            lines.append(f"    Travel time: {info['cost']:.4f}")

        lines.append("")
        lines.append(f"Nodes expanded across runs: {nodes_expanded}")

        self.route_output.insert(tk.END, "\n".join(lines))


def main():
    root = tk.Tk()
    ICS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
