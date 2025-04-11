import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor,VisionEncoderDecoderModel,Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from torchvision import transforms
from datasets import load_metric
from torch.cuda import is_available as cuda_available
from torchvision.transforms import ToTensor
import os
from PIL import Image



def load_model_with_check(model_name="microsoft/trocr-base-stage1", local_path="./saved_model"):
    # Create directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    try:
        # Try to load from local
        model = VisionEncoderDecoderModel.from_pretrained(local_path)
        processor = TrOCRProcessor.from_pretrained(local_path)
        print("Loaded model and processor from local storage")
        return model, processor
    except:
        # Download if local load fails
        print("Downloading model and processor...")
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name)
        
        # Save for future use
        model.save_pretrained(local_path)
        processor.save_pretrained(local_path)
        print("Saved model and processor to local storage")
        return model, processor












