import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

class PACSDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        """Init PACS dataset."""
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root_dir = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data, self.targets = self.load_images()
        

    def load_images(self):
        images, targets = [],[]
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_dir):
                image = Image.open(os.path.join(class_dir, image_name)).convert('RGB')
                label = self.class_to_idx[cls]
                images.append(image)
                targets.append(label)
        return images, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label