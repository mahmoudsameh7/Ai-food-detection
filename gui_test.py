import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

from prediction import class_name, predict_image, read_image


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "braintumor10Epoccategorical.h5"
PREVIEW_SIZE = (280, 280)


class BrainTumorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Brain Tumor Detection")
        self.root.geometry("520x650")
        self.root.minsize(480, 600)
        self.root.configure(bg="#f4f7fb")

        self.model = None
        self.preview_image = None
        self._build_styles()
        self._build_layout()
        self._load_model()

    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", background="#f4f7fb", foreground="#14213d", font=("Arial", 22, "bold"))
        style.configure("Subtitle.TLabel", background="#f4f7fb", foreground="#667085", font=("Arial", 10))
        style.configure("Status.TLabel", background="#ffffff", foreground="#344054", font=("Arial", 13, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#667085", font=("Arial", 10))
        style.configure("Primary.TButton", font=("Arial", 11, "bold"), padding=(16, 9))
        style.configure("Secondary.TButton", font=("Arial", 11), padding=(16, 9))

    def _build_layout(self):
        header = ttk.Frame(self.root, padding=(28, 24, 28, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Brain Tumor Detection", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Select an MRI image to receive a model prediction.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        card = tk.Frame(self.root, bg="#ffffff", highlightthickness=1, highlightbackground="#e4e7ec")
        card.pack(fill="both", expand=True, padx=28, pady=(10, 20))

        self.preview_panel = tk.Label(card, text="No image selected", bg="#eef2f6", fg="#98a2b3", font=("Arial", 12))
        self.preview_panel.pack(padx=24, pady=(28, 22), ipadx=20, ipady=20)

        self.status_label = ttk.Label(card, text="Loading model…", style="Status.TLabel", anchor="center")
        self.status_label.pack(fill="x", padx=24)
        self.hint_label = ttk.Label(card, text="Predictions are for assistance only and are not a medical diagnosis.", style="Hint.TLabel", anchor="center", wraplength=400)
        self.hint_label.pack(fill="x", padx=24, pady=(8, 22))

        actions = ttk.Frame(card)
        actions.pack(pady=(0, 28))
        self.choose_button = ttk.Button(actions, text="Choose image", style="Primary.TButton", command=self.select_image)
        self.choose_button.grid(row=0, column=0, padx=6)
        self.clear_button = ttk.Button(actions, text="Clear", style="Secondary.TButton", command=self.clear_image)
        self.clear_button.grid(row=0, column=1, padx=6)

    def _load_model(self):
        if not MODEL_PATH.is_file():
            self._set_status("Model file is missing", "#b42318")
            self.choose_button.state(["disabled"])
            return
        try:
            self.model = load_model(MODEL_PATH)
            self._set_status("Ready — choose an MRI image", "#344054")
        except Exception as exc:
            self._set_status("Could not load model", "#b42318")
            self.choose_button.state(["disabled"])
            messagebox.showerror("Model error", str(exc))

    def _set_status(self, text: str, color: str):
        self.status_label.configure(text=text, foreground=color)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Choose an MRI image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path or self.model is None:
            return

        image = read_image(Path(path))
        if image is None:
            messagebox.showerror("Image error", "The selected file could not be read as an image.")
            return

        try:
            preview = Image.open(path).convert("RGB")
            preview.thumbnail(PREVIEW_SIZE)
            self.preview_image = ImageTk.PhotoImage(preview)
            self.preview_panel.configure(image=self.preview_image, text="")
            class_id, confidence, _ = predict_image(self.model, Path(path))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        result = class_name(class_id).title()
        self._set_status(f"{result}  •  Confidence: {confidence:.1%}", "#027a48" if class_id == 0 else "#b42318")

    def clear_image(self):
        self.preview_image = None
        self.preview_panel.configure(image="", text="No image selected")
        self._set_status("Ready — choose an MRI image", "#344054")


def main():
    root = tk.Tk()
    BrainTumorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
