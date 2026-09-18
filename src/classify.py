"""ResNet-18 product-type classifier — inference only.

Shared by every notebook that needs to load the trained classifier and run
predictions with it (07_end_to_end_evaluation.ipynb, 08_pipeline_inference.ipynb,
10_active_learning.ipynb), so the loading logic can't drift between them.
Training itself lives in 06_a_model_classifier.ipynb, not here.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = 224

# ImageNet normalization, no augmentation — matches val_tf in
# 06_a_model_classifier.ipynb, since this is for inference, not training.
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet-18 with its final layer replaced to output num_classes logits.

    pretrained=True loads ImageNet weights, used only when training a fresh
    classifier (06_a). pretrained=False (the default here) is for inference,
    where a checkpoint's own weights fully replace them anyway.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(checkpoint_path: str, device: str = "cpu") -> tuple[nn.Module, list[str]]:
    """Load a trained classifier checkpoint.

    Returns (model, classes). classes always comes from the checkpoint's own
    'classes' key, never hardcoded here — a hardcoded list that drifts from
    the checkpoint has silently misrouted predictions before (see CLAUDE.md).
    """
    state = torch.load(checkpoint_path, map_location=device)
    classes = list(state["classes"])
    assert classes == sorted(classes), f"checkpoint classes not sorted: {classes}"
    model = build_model(num_classes=len(classes))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model.to(device), classes


def predict(
    model: nn.Module,
    img_path: str,
    classes: list[str],
    device: str = "cpu",
) -> tuple[str, float, np.ndarray]:
    """Return (predicted_class, confidence, full_softmax_probs) for a single image."""
    img = Image.open(img_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze().cpu().numpy()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), probs
