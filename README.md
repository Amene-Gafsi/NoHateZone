# NoHateZone

## File Structure
```text
├── README.md
│
├── data/
│   ├── HateMM/
│   │   └── HateMM_annotation_adjusted.xlsx: is the updated HateMM data including the modality- │   │       specific labels
│   └── MMHS150K/
│       └── Includes memes data and corresponding embeddings       
│
├── \texbf{main_process}/
│   ├── frame_audio_decomposer.py: decomposes the video into audio chunks and frames extracted │   │   per second
│   │    
│   ├── audio2text.py & utils.py: includes helper functions and model imports
│   │   
│   ├── chunk_frame_generator.py: generates the frames with 1-second frequency and extracts     │   │   the corresponding texts using OCR
│   │
│   ├── chunk_labeller_with_intervals.ipynb: gets the data ready to acquire the performance     │   │   metrics for our fine-tuned DistillBERT (DBertXhate)
│   │
│   └── .csv and .xslx files: are intermediary files to get the embeddings for model training
│
│
│
├── src/
│   ├── module1.py
│   └── module2.py
└── tests/
    └── test_module1.py
```

## Dataset Licenses

## Model Licenses


