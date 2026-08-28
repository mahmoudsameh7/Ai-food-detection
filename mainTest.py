import argparse
from pathlib import Path

from tensorflow.keras.models import load_model

from prediction import class_name, predict_image


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Classify a brain MRI image.")
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "braintumor10Epoccategorical.h5",
        help="Path to a trained Keras model.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = load_model(args.model)
    class_id, confidence, probabilities = predict_image(model, args.image)

    print("Raw prediction probabilities:", probabilities)
    print("Prediction result:", class_id)
    print(f"Predicted: {class_name(class_id)} ({confidence:.1%})")


if __name__ == "__main__":
    main()
