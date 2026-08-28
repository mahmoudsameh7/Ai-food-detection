from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model

INPUT_SIZE = 64


def read_image(path: Path):
    """Read an image safely, including paths with non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def prepare_image(path: Path):
    image = read_image(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model: Model, path: Path):
    probabilities = model.predict(prepare_image(path), verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    return class_id, float(probabilities[class_id]), probabilities


def class_name(class_id: int):
    return "NO TUMOR" if class_id == 0 else "TUMOR"
