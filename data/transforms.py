import numpy as np
import cv2
from typing import Tuple, Optional, Any, Dict
import random


class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Resize:
    """Resize image and mask to a fixed size."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        return img, mask


class RandomHorizontalFlip:
    """Horizontally flip the image and mask with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomVerticalFlip:
    """Vertically flip the image and mask with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.p:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class RandomRotation:
    """Randomly rotate the image and mask by 90, 180, or 270 degrees."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            img = np.rot90(img, k)
            if mask is not None:
                mask = np.rot90(mask, k)
            # Convert back to contiguous array to avoid stride issues
            img = np.ascontiguousarray(img)
            if mask is not None:
                mask = np.ascontiguousarray(mask)
        return img, mask


class Normalize:
    """Normalize image to [0, 1] or standard ImageNet stats."""
    def __init__(self, mean: Tuple[float, float, float] = (0.0, 0.0, 0.0), std: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img, mask


class ToTensor:
    """Convert HWC NumPy array to CHW PyTorch tensor (as NumPy for compatibility)."""
    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Image: HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # Mask: HW -> HW (no channel)
        return img, mask


class DefectAwareCrop:
    """
    Crop a high-res image while preserving anomalies.
    If mask is available and contains defects, ensure crop includes defect region.
    Otherwise, random crop.
    """
    def __init__(self, size: Tuple[int, int], p: float = 0.8):
        self.size = size  # (H, W)
        self.p = p  # probability to do defect-aware crop

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h_img, w_img = img.shape[:2]
        h_crop, w_crop = self.size

        if h_crop >= h_img or w_crop >= w_img:
            # If crop size >= image, return original
            return img, mask

        if mask is not None and random.random() < self.p:
            # Find bounding box of defect
            coords = np.column_stack(np.where(mask > 0))
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # Ensure crop includes the defect
                y1 = random.randint(max(0, y_max - h_crop + 1), min(y_min, h_img - h_crop))
                x1 = random.randint(max(0, x_max - w_crop + 1), min(x_min, w_img - w_crop))
            else:
                # No defect: random crop
                y1 = random.randint(0, h_img - h_crop)
                x1 = random.randint(0, w_img - w_crop)
        else:
            # Random crop
            y1 = random.randint(0, h_img - h_crop)
            x1 = random.randint(0, w_img - w_crop)

        img_crop = img[y1:y1 + h_crop, x1:x1 + w_crop]
        if mask is not None:
            mask_crop = mask[y1:y1 + h_crop, x1:x1 + w_crop]
        else:
            mask_crop = None

        return img_crop, mask_crop


class ColorJitter:
    """Randomly adjust brightness, contrast, and saturation."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = img.astype(np.float32)

        # Brightness
        alpha_b = 1.0 + random.uniform(-self.brightness, self.brightness)
        img = img * alpha_b

        # Contrast
        alpha_c = 1.0 + random.uniform(-self.contrast, self.contrast)
        img = img * alpha_c

        # Saturation
        alpha_s = 1.0 + random.uniform(-self.saturation, self.saturation)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img = alpha_s * img + (1 - alpha_s) * np.stack([gray, gray, gray], axis=-1)

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, mask


# Example usage
if __name__ == "__main__":
    # Simulate a high-res image (e.g., from VisA)
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[200:300, 400:500] = 255  # Simulate defect

    transform = Compose([
        DefectAwareCrop(size=(256, 256)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(p=0.5),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])

    img_t, mask_t = transform(img, mask)
    print(f"Transformed image shape: {img_t.shape}")
    if mask_t is not None:
        print(f"Transformed mask shape: {mask_t.shape}")