"""ResNet-18 product-type classifier — inference only."""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASSES = ['bullet_lipstick', 'liquid_lipstick', 'other', 'swatch']
IMG_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes=len(CLASSES))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model.to(device)


def predict(model: nn.Module, img_path: str, device: str = "cpu") -> tuple[str, float]:
    """Return (predicted_class, confidence) for a single image."""
    img = Image.open(img_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, idx = probs.max(dim=1)
    return CLASSES[idx.item()], conf.item()
