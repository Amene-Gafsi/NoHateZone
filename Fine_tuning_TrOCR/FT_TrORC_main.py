
import pandas as pd
import torch
from transformers import TrOCRProcessor,Seq2SeqTrainer, Seq2SeqTrainingArguments
from torchvision import transforms
from datasets import load_metric
from torch.cuda import is_available as cuda_available
from torchvision.transforms import ToTensor
#from FT_TrORC_utils import *
#from TrORC_initiator import configure_model, load_model_with_check
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
import argparse
from pathlib import Path
import sys



print(">>> sys.executable:", sys.executable)
print(">>> sys.path:", sys.path)
print(">>> PATH environment variable:", os.environ.get("PATH"))
print(">>> Which python is in PATH? ", os.popen('which python3').read())
try:
    import transformers
    print(">>> Transformers version:", transformers.__version__)
except Exception as e:
    print(">>> Error importing transformers:", e)
    raise

#---------------------------------------------DATASET CLASSES--------------------------------------------------#
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

#------------------------------------------------------------------------------------------------------#



#-------------------------------------------------MODEL CONSTRUCTOR FUNCTIONS------------------------------------------------#
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


def configure_model(model, processor, max_length=32, num_beams=5, 
                             no_repeat_ngram_size=2, length_penalty=0.8, 
                             temperature=0.9, do_sample=True):
    """
    Configure a TrOCR model for meme-style text generation.
    
    Args:
        model: TrOCR model instance
        processor: Processor with tokenizer
        max_length: Maximum output length (default: 32)
        num_beams: Number of beams for beam search (default: 5)
        no_repeat_ngram_size: Block repeating n-grams (default: 2)
        length_penalty: Length penalty (<1 = shorter, >1 = longer) (default: 0.8)
        temperature: Sampling temperature (default: 0.9)
        do_sample: Enable sampling (default: True)
    """
    # Text generation strategy
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = no_repeat_ngram_size
    model.config.length_penalty = length_penalty
    model.config.num_beams = num_beams

    # Token alignment
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Sampling parameters
    model.config.temperature = temperature
    model.config.do_sample = do_sample

    return model


# --- Command-Line Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--dataset_path", required=True, help="Base dataset directory")
    parser.add_argument("--results_path", required=True, help="Output directory")
    args = parser.parse_args()
    return args.dataset_path, args.results_path

#------------------------------------------------------------------------------------------------------#


DATASET_PATH, RESULTS_PATH = parse_args()

# Create results directory if it doesn't exist using pathlib
results_dir = Path(RESULTS_PATH)
results_dir.mkdir(parents=True, exist_ok=True)
# Optional: Create a checkpoints subdirectory
(results_dir / 'checkpoints').mkdir(exist_ok=True)

# --- Construct File and Directory Paths Relative to DATASET_PATH ---
# Define paths to your CSV files
#train_csv_path = os.path.join(DATASET_PATH, "processed", "MMF_HAR_comb_processed", "training", "mmf_har_train_combined.csv")
#test_csv_path  = os.path.join(DATASET_PATH, "processed", "MMF_HAR_comb_processed", "testing",  "mmf_har_test_combined.csv")

train_csv_path = os.path.join(DATASET_PATH, "data", "training", "mmf_har_train_combined.csv")
test_csv_path  = os.path.join(DATASET_PATH, "data", "testing",  "mmf_har_test_combined.csv")


# Load dataframes
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)


# Define the image directories
image_train_dir = os.path.join(DATASET_PATH, "data", "images", "training")
image_test_dir  = os.path.join(DATASET_PATH, "data", "images", "testing")



# Define the processor and model
# Load the TrOCR processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

train_dataset = MMF_HAR(root_dir=image_train_dir,
                           df=train_df,
                           processor=processor)
eval_dataset = MMF_HAR(root_dir=image_test_dir,
                           df=test_df,
                           processor=processor)



# Initiate the model and processor
pre_model, processor = load_model_with_check()



# Apply configuration
model = configure_model(
    model=pre_model,
    processor=processor,
    max_length=32,
    num_beams=5,
    temperature=0.9
)



training_args = Seq2SeqTrainingArguments(
    
    output_dir=str(results_dir / "checkpoints"),
    evaluation_strategy="steps",
    
    # Device optimization
    fp16=cuda_available(),  # Only enable if CUDA is available
    
    # Adjusted steps for 3500 training samples
    per_device_train_batch_size=8 if cuda_available() else 2,
    per_device_eval_batch_size=8 if cuda_available() else 2,
    eval_steps=100,
    save_steps=200,
    
    # Training parameters
    learning_rate=3e-5,
    num_train_epochs=15,

    # Phase 1: (Learning rate warmup) increases linearly from 0 to 3e-5
    # Phase 2: (Learning rate decay) decreases linearly from 3e-5 to 0
    warmup_steps=50,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    
    
    # Generation & metrics
    predict_with_generate=True,
    generation_max_length=32,  # Match meme text length
    
    
    # Model saving
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,

    report_to= "none"
    
)


#Defining the CER metric for evaluation
cer_metric = load_metric("cer")



def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}




#Define Augmentations
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=(3,7), sigma=(0.1, 0.5))
])

#Create Augmented Datasets
train_dataset = AugmentedMMFHAR(
    root_dir=image_train_dir,
    df=train_df,
    processor=processor,
    transform=train_transform,
    max_target_length=32
)

eval_dataset = AugmentedMMFHAR(  # No augmentations for eval
    root_dir=image_test_dir,
    df=test_df,
    processor=processor,
    transform=None, 
    max_target_length=32
)



# Update Trainer with Custom Collator
def augmented_collator(batch):
    to_tensor = ToTensor()  # Create converter inside collator
    
    # Convert all images to tensors
    pixel_values = torch.stack([
        item["pixel_values"] if torch.is_tensor(item["pixel_values"]) 
        else to_tensor(item["pixel_values"]) 
        for item in batch
    ])
    
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# Instantiate Trainer with Augmentations
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=augmented_collator,  # Use custom collator
)

print("Starting training...")
trainer.train()
print("Training completed.")































