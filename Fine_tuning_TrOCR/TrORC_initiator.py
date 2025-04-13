
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
from torch.cuda import is_available as cuda_available
import os




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






