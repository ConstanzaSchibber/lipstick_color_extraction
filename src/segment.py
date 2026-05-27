"""U-Net color-region segmenter — inference only."""
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image

IMG_SIZE = 256

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model() -> torch.nn.Module:
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def load_segmenter(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    model = build_model()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model.to(device)


def predict_mask(
    model: torch.nn.Module,
    img_path: str,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """Return binary mask (H x W, values 0/1) resized to the original image size."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    return np.array(Image.fromarray(mask * 255).resize((orig_w, orig_h), Image.NEAREST)) // 255
