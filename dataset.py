import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import resize_x, resize_y, batchsize


class FlowerDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.ToTensor()
        ])

        # Get class folders (sorted for consistency)
        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Load image paths and labels
        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Only include image files
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {len(self.class_names)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]

        return image, label


def flowerLoader(root_dir):
    dataset = FlowerDataset(root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=2  # can set 0 if errors on Windows
    )

    return loader
