
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.cuda import is_available as cuda_available





class MMF_HAR(Dataset):

    def __init__(self, root_dir, df, processor, max_target_length=32):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)  # Important after splits
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data from dataframe
        image_name = self.df.iloc[idx]['image']
        text = self.df.iloc[idx]['text']
        
        # Load and process image
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Process text
        labels = self.processor.tokenizer(
            text,
            padding="max_length", #Pad shorter texts to 'max_target_length'
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Replace padding token with -100 to train the decoder part of the model
        # This is important for the loss function to ignore padding tokens in the labels during training
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 
                 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }


# Modify Dataset Class
class AugmentedMMFHAR(MMF_HAR):
    
    def __init__(self, root_dir, df, processor, transform=None, **kwargs):
        super().__init__(root_dir, df, processor, **kwargs)
        self.transform = transform
        
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Apply augmentations only to training images
        if self.transform:
            image = transforms.ToPILImage()(item["pixel_values"])
            item["pixel_values"] = transforms.ToTensor()(self.transform(image))
            
        return item











