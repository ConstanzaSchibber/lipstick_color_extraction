"""Batched inference for the trained classifier, for scoring many images at once.

src/classify.py's predict() loads and transforms one image at a time -- the
right shape for the single-image lookups 07_end_to_end_evaluation.ipynb and
src/pipeline.py's route_and_extract need. This module is for the opposite
case: classifying an entire corpus (all ~9k product images in
08_pipeline_inference.ipynb, and any future notebook doing the same
full-corpus scoring) through a DataLoader instead of one image at a time.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from src.classify import IMG_SIZE

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class _ImagePathDataset(Dataset):
    """Loads and transforms images by path, tolerating unreadable files.

    Returns a zero tensor (rather than raising) for a path PIL can't open,
    so one corrupt image doesn't halt inference over the rest of the corpus.
    """
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            return _transform(Image.open(self.paths[i]).convert("RGB")), i
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), i


def predict_batch(
    model: nn.Module,
    paths: list[str],
    classes: list[str],
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    desc: str = "images classified",
) -> tuple[list[str], list[float]]:
    """Classify every image in paths, batched through model.

    Returns (pred_labels, confidences), aligned to paths by position.
    """
    loader = DataLoader(_ImagePathDataset(paths), batch_size=batch_size, num_workers=num_workers)
    idx2pred = {}

    with torch.no_grad():
        for batch, indices in tqdm(loader, desc=desc, unit="batch"):
            probs = torch.softmax(model(batch.to(device)), dim=1).cpu().numpy()
            for prob, idx in zip(probs, indices.numpy()):
                idx2pred[int(idx)] = (classes[prob.argmax()], float(prob.max()))

    pred_labels = [idx2pred[i][0] for i in range(len(paths))]
    confidences = [idx2pred[i][1] for i in range(len(paths))]
    return pred_labels, confidences
