# Import packages
import numpy as np
import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

# Custom dataset class
class ImageDataset(Dataset):

    def __init__(self, data_dir, dataset_type='train', train_size=0.85):

        # Initialize variables
        self.file_dirs = []
        self.labels = []
        self.num_obs = {}
        self.label_names = {}
        
        j=0
        # Load the dataset according to directory structure
        for sub in os.listdir(data_dir):
            sub_dir = os.path.join(data_dir, sub)
            for subsub in os.listdir(sub_dir):
                subsub_dir = os.path.join(sub_dir, subsub)
                num_class = len(os.listdir(subsub_dir))

                # Load the images
                for i, file in enumerate(os.listdir(subsub_dir)):
                    if dataset_type == 'train' and i < train_size*num_class:
                        self.file_dirs.append(os.path.join(subsub_dir, file))
                        self.labels.append(j)
                    elif dataset_type == 'val' and i >= train_size*num_class:
                        self.file_dirs.append(os.path.join(subsub_dir, file))
                        self.labels.append(j)

                # Extract the label names and number of observations
                self.label_names[j] = subsub
                self.num_obs[j] = np.sum(np.array(self.labels) == j)
                j += 1

        # Define the transformations
        if dataset_type == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.file_dirs)

    def __getitem__(self, idx):

        # Get correct file path and label
        file_path = self.file_dirs[idx]
        label = self.labels[idx]

        # Load the image
        image = read_image(file_path)
        image = transforms.ToPILImage()(image)

        # Apply transformations
        image = self.transform(image)

        return image, label