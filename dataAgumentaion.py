import torch
from torchvision import transforms

# Data Augmentation for the training set
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(p=0.5),        # Randomly flip the image vertically
    transforms.RandomRotation(15),               # Randomly rotate the image by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly change brightness, contrast, saturation, and hue
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Wrap datasets with data augmentation and normalization
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, ID, gender, hand, finger = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, ID, gender, hand, finger