from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


MODEL_PATH=Path(__file__).resolve().with_name("banana_model.pt")
LABELS=("overripe", "ripe", "rotten", "unripe")
DAY_GUIDE={
    "unripe": "5-7 days",
    "ripe": "2-3 days",
    "overripe": "0-1 day",
    "rotten": "0 days; discard",
}
IMAGE_SIZE=64


class banana_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, len(LABELS)),
        )

    def forward(self, image):
        return self.classifier(self.features(image))


def image_tensor(image):
    image=image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array=np.asarray(image, dtype=np.float32) / 255.0
    tensor=torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def has_ripe_color_evidence(image):
    sample=image.convert("RGB").resize((128, 128))
    hsv=np.asarray(sample.convert("HSV"), dtype=np.uint8)
    hue, saturation, value=hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow=(hue >= 8) & (hue <= 48) & (saturation >= 35) & (value >= 70)
    if not yellow.any():
        yellow=(saturation >= 35) & (value >= 70)
    if not yellow.any():
        return False

    yellow_brightness=float(value[yellow].mean()) / 255
    brown_fraction=float(
        ((hue >= 5) & (hue <= 35) & (value < 175) & yellow).sum()
    ) / int(yellow.sum())
    dark_fraction=float(((value < 90) & yellow).sum()) / int(yellow.sum())
    return (
        float(yellow.mean()) >= 0.30
        and yellow_brightness >= 0.80
        and brown_fraction <= 0.20
        and dark_fraction <= 0.05
    )


def predict_banana(image_path):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    image_path=Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError("Choose a JPG, PNG, BMP, or WEBP banana image.")
    if image_path.stat().st_size > 15 * 1024 * 1024:
        raise ValueError("Image is too large; choose a file smaller than 15 MB.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Banana model not found at {MODEL_PATH}. Run all cells in banana.ipynb first."
        )

    try:
        with Image.open(image_path) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("Image dimensions are too large to process.")
            tensor=image_tensor(image).unsqueeze(0)
            ripe_color_evidence=has_ripe_color_evidence(image)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read the selected image: {error}") from error

    artifact=torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("The saved banana model artifact is invalid.")
    labels=artifact.get("labels")
    if not isinstance(labels, list) or tuple(labels) != LABELS:
        raise ValueError("The saved banana model has an invalid class schema.")
    model=banana_cnn()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities=torch.softmax(model(tensor), dim=1)[0]
    index=int(probabilities.argmax())
    used_color_correction=labels[index] == "rotten" and ripe_color_evidence
    if used_color_correction:
        index=labels.index("ripe")
    label=labels[index]
    warning=(
        "Ripeness class only; days are a rough room-temperature guide, not a "
        "measured expiry date. Storage conditions can change this; model confidence "
        "is not calibrated. Use banana-only images; other objects can be misclassified."
    )
    if used_color_correction:
        warning=(
            "The CNN favored rotten, but strong bright-yellow color with little "
            "brown or dark area triggered a conservative ripe correction. Please "
            "visually confirm the result. The displayed score is the CNN score for "
            "ripe, not a calibrated probability. " + warning
        )
    return {
        "label": label,
        "confidence": float(probabilities[index]),
        "days_left": DAY_GUIDE[label],
        "warning": warning,
    }
