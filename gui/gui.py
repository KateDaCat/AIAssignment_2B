import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import sys
import os

# Routing imports (from your existing A2A code)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import load_graph
from algorithms.dfs import dfs_search
from algorithms.bfs import bfs_search
from algorithms.cus1_ucs import cus1_ucs
from algorithms.astar import astar_search
from algorithms.gbfs import gbfs_search
from algorithms.cus2_hcs import cus2_hcs

# =============================
# ICS GUI (Assignment 2B)
# =============================
class ICS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ICS – Incident Classification System")
        self.root.geometry("1350x750")

        # ML prediction
        self.uploaded_img = None
        self.cnn_model = None
        self.model2 = None

        self.build_layout()

    # ---------------------------
    # Main UI layout
    # ---------------------------
    def build_layout(self):

        # FRAME LEFT = Routing + Map
        left = tk.Frame(self.root, width=850, height=750, bg="white")
        left.pack(side="left", fill="both", expand=True)

        self.map_canvas = tk.Canvas(left, bg="white", highlightbackground="#ccc")
        self.map_canvas.pack(fill="both", expand=True)

        # FRAME RIGHT = AI classification
        right = tk.Frame(self.root, width=500, bg="#f5f5f5", padx=20, pady=20)
        right.pack(side="right", fill="y")

        tk.Label(right, text="Accident Image Classification",
                 font=("Arial", 16, "bold")).pack(pady=10)

        # Image preview box
        self.preview = tk.Label(right, text="No Image Uploaded",
                                bg="#ddd", width=40, height=12)
        self.preview.pack(pady=10)

        tk.Button(right, text="Upload Image",
                  command=self.upload_image,
                  width=25).pack(pady=5)

        # Predictions
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

        # Routing Panel
        tk.Label(right, text="Route Finder", font=("Arial", 16, "bold")).pack()

        tk.Button(right, text="Run Routing",
                  command=self.run_routing,
                  width=25).pack(pady=10)

        self.route_output = tk.Text(right, height=12, width=50)
        self.route_output.pack()

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
    def run_routing(self):

        # TODO: load graph, call your algorithms, draw map
        # Use your existing 1A code here

        # Example output:
        self.route_output.delete("1.0", tk.END)
        self.route_output.insert(tk.END,
            "Origin: X\nDestination: Y\nBest Routes:\n1) Path A → B → C (12 min)"
        )


def main():
    root = tk.Tk()
    ICS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
