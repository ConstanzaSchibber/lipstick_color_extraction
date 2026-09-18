"""U-Net color-region segmenter — inference only.

One architecture (ResNet-18 encoder, single output channel) is shared by all
four trained segmenters — bullet/liquid/pencil, closed, swatch, and lips (see
06_b_model_segmenters.ipynb) — only the checkpoint weights differ, so
load_segmenter's checkpoint_path argument is the only thing that changes
between them. Shared by every notebook that needs segmenter predictions
(07_end_to_end_evaluation.ipynb, 08_pipeline_inference.ipynb).
"""
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
    """ResNet-18-encoder U-Net with a single output channel: a binary
    color-region mask. encoder_weights=None since this is inference-only —
    the checkpoint's own weights fully replace them anyway."""
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def load_segmenter(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load one of the four trained segmenter checkpoints (see
    06_b_model_segmenters.ipynb for how each was trained)."""
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
    """Predict a binary mask (H x W, values 0/1) for img_path, resized back to
    the original image's dimensions (the model itself always outputs a fixed
    IMG_SIZE x IMG_SIZE mask)."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    return np.array(Image.fromarray(mask * 255).resize((orig_w, orig_h), Image.NEAREST)) // 255
