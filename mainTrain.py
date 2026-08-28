import argparse
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.utils import to_categorical


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_SIZE = 64
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_class_dirs(root: Path):
    """Find a directory containing the expected `no` and `yes` classes."""
    for no_dir in root.rglob("no"):
        yes_dir = no_dir.parent / "yes"
        if no_dir.is_dir() and yes_dir.is_dir():
            return no_dir, yes_dir
    return None


def find_or_extract_dataset(dataset_path: Path):
    """Return the class directories, extracting the bundled ZIP when needed."""
    found = find_class_dirs(dataset_path) if dataset_path.is_dir() else None
    if found:
        return found

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        extract_dir = PROJECT_DIR / "data"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(dataset_path) as archive:
            archive.extractall(extract_dir)
        found = find_class_dirs(extract_dir)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find dataset folders named 'no' and 'yes'. "
        f"Checked: {dataset_path}"
    )


def imread_unicode(path: Path):
    """Read image paths containing non-ASCII characters on all platforms."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def load_folder(folder: Path, value: int, dataset: list, labels: list):
    skipped = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
            continue
        image = imread_unicode(path)
        if image is None:
            skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.asarray(image, dtype=np.float32) / 255.0)
        labels.append(value)
    return skipped


def build_model():
    model = Sequential(
        [
            Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
            Conv2D(32, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), kernel_initializer="he_uniform"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), kernel_initializer="he_uniform"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64),
            Activation("relu"),
            Dropout(0.5),
            Dense(2),
            Activation("softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the brain tumor classifier.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIR / "brain_tumor_datase.zip",
        help="Dataset folder or ZIP file containing class folders named no and yes.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "braintumor10Epoccategorical.h5")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    no_dir, yes_dir = find_or_extract_dataset(args.dataset)
    dataset, labels = [], []
    skipped_no = load_folder(no_dir, 0, dataset, labels)
    skipped_yes = load_folder(yes_dir, 1, dataset, labels)

    if len(dataset) < 2 or len(set(labels)) < 2:
        raise ValueError("The dataset must contain readable images in both 'no' and 'yes' folders.")

    dataset = np.asarray(dataset, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    print(f"Loaded: {len(dataset)} images | Skipped unreadable: no={skipped_no}, yes={skipped_yes}")

    x_train, x_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.2, random_state=0, stratify=labels
    )
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = build_model()
    model.fit(
        x_train,
        y_train,
        batch_size=16,
        verbose=1,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
