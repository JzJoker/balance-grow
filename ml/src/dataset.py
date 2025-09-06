import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

class HairSegDataset(Dataset):
    """
    PyTorch Dataset for hair segmentation
    Returns (image_tensor, mask_tensor)
    """
    def __init__(self, images_root, masks_root, transform=None):
        self.images_root = images_root
        self.masks_root = masks_root
        self.transform = transform

        # Collect all image paths
        self.images = [
            os.path.join(images_root, f) for f in os.listdir(images_root)
            if f.endswith(('.png', '.jpg'))
        ]
        self.images.sort()  # ensures consistent ordering

        # Pre-cache mask paths for faster lookup
        self.image_to_masks = {}  # {image_number: {"hair": path, "skin": path}
        for folder in os.listdir(masks_root):
            folder_path = os.path.join(masks_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for f in os.listdir(folder_path):
                if f.endswith(".png"):
                    img_number = f.split("_")[0].zfill(5)  # pad to 5 digits
                    if img_number not in self.image_to_masks:
                        self.image_to_masks[img_number] = {}
                    if "_hair" in f:
                        self.image_to_masks[img_number]["hair"] = os.path.join(folder_path, f)
                    elif "_skin" in f:
                        self.image_to_masks[img_number]["skin"] = os.path.join(folder_path, f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_number = os.path.splitext(os.path.basename(img_path))[0].zfill(5)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Create combined mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_paths = self.image_to_masks.get(img_number, {})

        

        # Hair = 2
        hair_path = mask_paths.get("hair")
        if hair_path and os.path.exists(hair_path):
            hair_mask = cv2.imread(hair_path, cv2.IMREAD_GRAYSCALE)
            hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask[hair_mask > 0] = 2

        # Skin = 1 (overwrite background, but not hair)
        skin_path = mask_paths.get("skin")
        if skin_path and os.path.exists(skin_path):
            skin_mask = cv2.imread(skin_path, cv2.IMREAD_GRAYSCALE)
            skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask[(skin_mask > 0) & (mask == 0)] = 1

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
# Example Albumentations transforms
def get_default_transforms():
    return Compose([
        RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])

# Quick test
if __name__ == "__main__":
    dataset = HairSegDataset("../data/images", "../data/masks", transform=get_default_transforms())
    print(f"Dataset length: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}, Mask unique values: {np.unique(mask.numpy())}")
