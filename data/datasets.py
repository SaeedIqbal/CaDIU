import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class IndustrialAnomalyDataset(Dataset):
    """
    Base class for industrial anomaly detection datasets.
    Supports VisA, MVTec-AD, Real-IAD, and BTAD.
    """
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str = "train",  # 'train', 'test', or 'all'
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        task_type: str = "all"  # e.g., 'screw', 'capsule', or 'all'
    ):
        """
        Args:
            root_dir (str): Root directory containing all datasets (e.g., '/home/phd/datasets/')
            dataset_name (str): One of ['visa', 'mvtec', 'real_iad', 'btad']
            split (str): 'train', 'test', or 'all'
            transform (Callable, optional): Transform for input images
            target_transform (Callable, optional): Transform for masks
            task_type (str): Specific object category or 'all'
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.task_type = task_type.lower()
        self.transform = transform
        self.target_transform = target_transform

        if self.dataset_name not in ["visa", "mvtec", "real_iad", "btad"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load list of (image_path, mask_path, label) tuples."""
        samples = []

        if self.dataset_name == "visa":
            samples = self._load_visa()
        elif self.dataset_name == "mvtec":
            samples = self._load_mvtec()
        elif self.dataset_name == "real_iad":
            samples = self._load_real_iad()
        elif self.dataset_name == "btad":
            samples = self._load_btad()

        return samples

    def _load_visa(self) -> List[Dict[str, str]]:
        base_path = self.root_dir / "VisA"
        if not base_path.exists():
            raise FileNotFoundError(f"VisA dataset not found at {base_path}")

        categories = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
        if self.task_type != "all" and self.task_type not in categories:
            raise ValueError(f"Category '{self.task_type}' not in VisA. Available: {categories}")

        selected_categories = [self.task_type] if self.task_type != "all" else categories
        samples = []

        for cat in selected_categories:
            cat_path = base_path / cat
            split_dir = "1cls" if (cat_path / "1cls").exists() else "1class"  # handle variant
            data_path = cat_path / split_dir

            if not data_path.exists():
                continue

            # Train: only good images
            if self.split in ["train", "all"]:
                train_good = data_path / "train" / "good"
                if train_good.exists():
                    for img_path in sorted(train_good.glob("*.JPG")):
                        samples.append({
                            "image": str(img_path),
                            "mask": "",  # no mask for normal
                            "label": 0,
                            "category": cat
                        })

            # Test: good + anomaly
            if self.split in ["test", "all"]:
                test_path = data_path / "test"
                if test_path.exists():
                    for defect_type in test_path.iterdir():
                        if not defect_type.is_dir():
                            continue
                        for img_path in sorted(defect_type.glob("*.JPG")):
                            mask_path = str(img_path).replace("test", "ground_truth").replace(".JPG", "_mask.png")
                            has_anomaly = defect_type.name != "good"
                            samples.append({
                                "image": str(img_path),
                                "mask": mask_path if has_anomaly else "",
                                "label": int(has_anomaly),
                                "category": cat
                            })
        return samples

    def _load_mvtec(self) -> List[Dict[str, str]]:
        base_path = self.root_dir / "mvtec_anomaly_detection"
        if not base_path.exists():
            raise FileNotFoundError(f"MVTec-AD dataset not found at {base_path}")

        categories = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
        if self.task_type != "all" and self.task_type not in categories:
            raise ValueError(f"Category '{self.task_type}' not in MVTec-AD. Available: {categories}")

        selected_categories = [self.task_type] if self.task_type != "all" else categories
        samples = []

        for cat in selected_categories:
            cat_path = base_path / cat

            if self.split in ["train", "all"]:
                train_good = cat_path / "train" / "good"
                if train_good.exists():
                    for img_path in sorted(train_good.glob("*.png")):
                        samples.append({
                            "image": str(img_path),
                            "mask": "",
                            "label": 0,
                            "category": cat
                        })

            if self.split in ["test", "all"]:
                test_path = cat_path / "test"
                if test_path.exists():
                    for defect_type in sorted(test_path.iterdir()):
                        if not defect_type.is_dir():
                            continue
                        for img_path in sorted(defect_type.glob("*.png")):
                            mask_path = str(img_path).replace("test", "ground_truth").replace(".png", "_mask.png")
                            has_anomaly = defect_type.name != "good"
                            samples.append({
                                "image": str(img_path),
                                "mask": mask_path if has_anomaly else "",
                                "label": int(has_anomaly),
                                "category": cat
                            })
        return samples

    def _load_real_iad(self) -> List[Dict[str, str]]:
        base_path = self.root_dir / "Real-IAD"
        if not base_path.exists():
            raise FileNotFoundError(f"Real-IAD dataset not found at {base_path}")

        meta_file = base_path / "real_iad.json"
        if not meta_file.exists():
            raise FileNotFoundError("Real-IAD metadata (real_iad.json) not found")

        with open(meta_file, 'r') as f:
            meta = json.load(f)

        categories = list(meta["categories"].keys())
        if self.task_type != "all" and self.task_type not in categories:
            raise ValueError(f"Category '{self.task_type}' not in Real-IAD. Available: {categories}")

        selected_categories = [self.task_type] if self.task_type != "all" else categories
        samples = []

        for cat in selected_categories:
            cat_info = meta["categories"][cat]
            img_dir = base_path / cat_info["image_dir"]
            mask_dir = base_path / cat_info.get("mask_dir", "")

            for img_rel_path in cat_info["images"]:
                img_path = img_dir / img_rel_path
                if not img_path.exists():
                    continue

                label = 1 if cat_info["images"][img_rel_path]["anomaly"] else 0
                mask_path = ""
                if label == 1 and mask_dir:
                    mask_path = str(mask_dir / img_rel_path.replace(".jpg", "_mask.png"))

                # Filter by split if metadata includes it
                split_flag = cat_info["images"][img_rel_path].get("split", "test")
                if self.split == "train" and split_flag != "train":
                    continue
                if self.split == "test" and split_flag != "test":
                    continue

                samples.append({
                    "image": str(img_path),
                    "mask": mask_path,
                    "label": label,
                    "category": cat
                })
        return samples

    def _load_btad(self) -> List[Dict[str, str]]:
        base_path = self.root_dir / "BTAD"
        if not base_path.exists():
            raise FileNotFoundError(f"BTAD dataset not found at {base_path}")

        categories = ["01", "02", "03"]  # BTAD uses numeric IDs
        if self.task_type != "all" and self.task_type not in categories:
            raise ValueError(f"Category '{self.task_type}' not in BTAD. Available: {categories}")

        selected_categories = [self.task_type] if self.task_type != "all" else categories
        samples = []

        for cat in selected_categories:
            cat_path = base_path / cat

            if self.split in ["train", "all"]:
                train_good = cat_path / "train" / "good"
                if train_good.exists():
                    for img_path in sorted(train_good.glob("*.bmp")):
                        samples.append({
                            "image": str(img_path),
                            "mask": "",
                            "label": 0,
                            "category": cat
                        })

            if self.split in ["test", "all"]:
                test_path = cat_path / "test"
                if test_path.exists():
                    for defect_type in sorted(test_path.iterdir()):
                        if not defect_type.is_dir():
                            continue
                        for img_path in sorted(defect_type.glob("*.bmp")):
                            mask_path = str(img_path).replace("test", "ground_truth").replace(".bmp", ".bmp")
                            has_anomaly = defect_type.name != "good"
                            samples.append({
                                "image": str(img_path),
                                "mask": mask_path if has_anomaly else "",
                                "label": int(has_anomaly),
                                "category": cat
                            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, str]:
        item = self.samples[idx]
        image = cv2.imread(item["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if item["mask"] and os.path.exists(item["mask"]):
            mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask, item["label"], item["category"]


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = IndustrialAnomalyDataset(
        root_dir="/home/phd/datasets/",
        dataset_name="visa",
        split="test",
        transform=transform,
        task_type="all"
    )

    print(f"Loaded {len(dataset)} samples from VisA.")
    img, mask, label, cat = dataset[0]
    print(f"Sample: image shape={img.shape}, mask shape={mask.shape}, label={label}, category={cat}")