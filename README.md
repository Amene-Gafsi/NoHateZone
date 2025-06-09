# NoHateZone

## File Structure
```text
│
│
├── README.md
│
├── data/
│   ├── HateMM/
│   │   └── HateMM_annotation_adjusted.xlsx: is the updated HateMM data including the modality-specific labels
│   └── MMHS150K/
│       └── Includes memes data and corresponding embeddings       
│
│
├── main_process/
│   ├── frame_audio_decomposer.py: decomposes the video into audio chunks and frames extracted per second
│   │    
│   ├── audio2text.py & utils.py: includes helper functions and model imports
│   │   
│   ├── chunk_frame_generator.py: generates the frames with 1-second frequency and extracts the corresponding texts using OCR
│   │
│   ├── chunk_labeller_with_intervals.ipynb: gets the data ready to acquire the performance metrics for our fine-tuned DistillBERT (DBertXhate)
│   │
│   └── .csv and .xslx files: intermediary files to get the embeddings for model training
│
│
├── src/
│   ├── embedding/
│   │   ├── audio_image_combiner.ipnyb: selects a random image given the audio chunk intervals
│   │   │
│   │   └── other .py files: get the embeddings (768 x 1) for HateMM and MMHS150K using our ViT and DistillBERT
│   │
│   ├── models/   
│   │   ├── crossval_model.py:
│   │   │
│   │   └── fusion_model.py:
│   │
│   ├── tests/
│   │   └──
│   │
│   ├── train/
│   │   ├── bert_finetune.py: training file for DistillBERT fine-tuning using the HateSpeechDataset for detecting hate in audio transcriptions
│   │   │
│   │   ├── crossval_train.py:
│   │   │
│   │   ├── fine_tune_SCMA.py: SCMA fusion architecture fine-tuned on HateMM data (audio + image + OCR text)
│   │   │
│   │   ├── finetune_fusion.py: DCMA fusion architecture fine-tuned on HateMM data (image + OCR text)
│   │   │
│   │   ├── fusion_arch_v3.py: SCMA fusion architecture pre-trained on memes dataset (MMHS150K)
│   │   │
│   │   ├── train_fusion.py: DCMA fusion architecture pre-trained on memes dataset (MMHS150K)
│   │   │
│   │   └── random_classifier.py: Random classifier tested on HateMM data (image + OCR text)
│   │   
│   │     
│   ├── evaluation/
│   │   ├── SCMA_fine-tune_results/
│   │   │   ├── result_analyzer.py: calcualtes the standard deviation of the metrics in text_metrics.csv
│   │   │   │  
│   │   │   └── test_metrics.csv and roc_per_seed.png: results of SCMA fine-tuning 
│   │   ├── SCMA_fine-tune_results/
│   │   │   └── results of SCMA pre-training
│   │   │
│   │   └── other .py files for evaluation purposes
│   │
│   ├── utils/
│       └── includes the helper functions
│
│
├── media/
│   ├── 'input/video'/
│   │    └── includes the input video for inference
│   └── 'output/video'/
│       └── includes the censored video  
│
│
├── submission/
│   └── includes the necessary files for project submission
│
│


```

## Dataset Licenses

HateMM license:
https://creativecommons.org/licenses/by/4.0/legalcode
HateXplain license:
https://github.com/hate-alert/HateXplain?tab=MIT-1-ov-file

## Model Licenses


