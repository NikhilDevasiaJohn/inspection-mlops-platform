import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

CLASS_NAMES = ["good", "crack", "faulty_imprint", "scratch", "contamination", "color", "combined"]

class MVTecPillDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224):
        self.root = Path(root_dir) / "pill" / split
        self.img_size = img_size
        self.samples = []
        self.labels = []

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        for label_idx, class_name in enumerate(sorted(os.listdir(self.root))):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append(img_path)
                self.labels.append(0 if class_name == "good" else 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]