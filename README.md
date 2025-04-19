# NoHateZone

## **Our idea**
Use re-annotated meme, texts(tweets), audio speech and video datasets to ‘beep’ out audio chunks and blur out frames in videos where hate speech occurs.
The selected audio chunks and frames are going to fed into a simple neural network whose objective is to classify the object as the source of the hate speech: Race, Ethnicity, Religion and other.

## **Introduction**
Social media comprises platforms that allow users to connect with others to expand their knowledge, have a laugh, and discuss issues with global impacts. For social media to serve these purposes, it is essential to foster a safe environment where people can share content and interact without offending, humiliating, or bullying others. According to statistics shared by the FBI’s Uniform Crime Reporting Program, hate crimes occur in three main contexts: gender/sexual orientation, religion, and race/ethnicity. Together, these constitute 96% of the hate crimes reported in the United States in 2024. While social media enables a broad spectrum of self-expression across various topics, hate speech is not defined by its subject matter alone. Rather, hate speech is characterized by language that is hateful, discriminatory, or incites harm (journals.sagepub.com). This distinction underscores that although hate speech may appear in diverse forms, its defining criteria set it apart from general self-expression.

## **Contribution to on-going research**
Researches in the field of hate speech classification focuses on the end problem of classifying a video as hateful as not hateful. As in article (3), the authors proposed a deep learning framework to classify videos as hateful or not hateful by combining state-of-the-art transformers with a fusion layer. Although it is a novel approach, it doesn’t generate any addiitonal insight on the reasoning behind the classification. Our deep learning framework attempts to explain the full methodology behind the classification by providing frame-level and chunk-level rationale for audio component of a video.


## **Potential data sources**

### ***Text (tweets)***

 - **HateXplain dataset**
     - Contains word-level annotations: a word w is assigned a rationale of 1 if it contributes to the classification of the sentence as hate speech, and 0 otherwise

### ***Memes***
Article (1) states that In order to capture temporal dynamics of videos in classification problems, it is crucial to consider the three main modalities which are text, image and audio. As memes inherently combine visual and textual elements, they closely mirror the multimodal structure of videos. Therefore memes become a go-to choice to augment video-based hate speech datasets as they remain signficantly smaller compared to datasets beloning to other modalities.

 - **Facebook Hateful Memes dataset(FHM)**

 - **Multimedia automatic misogyny Identification(MAMI)**

 - **Re-annoted meme dataset**

### ***Videos***

 - **MultiHateClip (MHC)**

 - **HateMM(HMM)**

## **Research**

### **Cross-Modal Transfer from Memes to Videos: Addressing Data Scarcity in Hateful Video Detection (1)**

 - The vision modality requires different handling, as videos consist of multiple frames, while memes are represented by single images.
 - For video-based VLMs, such as LLaVA-NeXT-Video-7B, which can process multiple frames, we sample 16 frames per video, aligning with the model author’s recommendations and the average length of our video dataset
 - To adapt meme data for video-based VLMs, we simulate a video format by applying random image augmentations (e.g., rotation, cropping, etc.) to generate 16 frames from a single meme image. We employ Low-Rank Adaptation (LoRA) adapters to fine-tune the pre-trained VLM during.

### **Hate Speech recognition in Audio dataset (2)**
 - In this study, we combine both approaches to classify entire audio clips as containing hate speech or normal speech, while also pinpointing the exact segments within the audio where hate speech occurs, using a 10-millisecond time grid.
 - Cascading and End-to-End (E2E)
     - pinpoint hate speech within specific time frames

### **HateMM: A Multi-Modal Dataset for Hate Video Classification (3)**
 - BitChute- labels along with the frame spans 
 - Image-based hate detection methods cannot be directly adapted
 - Combined BERT-VIT-MFCC with the help of fusion layer

### **Detecting and Understanding Harmful Memes: A Survey**
Memes typically consist of images containing some text [Shifman, 2013;
Suryawanshi et al., 2020a,b].

**Hateful meme definition**: multimodal units consisting of an image and
embedded text that have the potential to cause harm to an individual, an organization, a community, or society in general

### **TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models**
 - TrOCR consists of an image Transformer encoder and an autoregressive text Transformer
 - The model can be **fine-tuned** with human-label datasets
 - **Optical Character Recognition (OCR)** is the electronic or mechanical conversion of images of types, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene photo or from subtitle text superimposed on an image

![image](https://github.com/user-attachments/assets/88dd0fe7-6849-487e-b53a-7dea758421e7)

 - TrOCR does not require any convolutional network for the backbone and does not introduce any image-specific inductive biases which make the model very easy to implement and maintain.

**Architecture**


### **An Investigation Into Explainable Audio Hate Speech Detection (4)**


### **Multimodal hate speech detection via multi-scale visual kernels and knowledge distillation architecture (5)**

### **Flamingo: a Visual Language Model for Few-Shot Learning**
 - **Flamingo** models can proces arbitrarily interleaved visual data and text input
 - **Flamingo** models can also be effectively **fine-tuned**
 - Though the model only directly attends to a single image at a time, the dependency on all previous images remains via self-attention in the LM. This single-image cross-attention scheme importantly allows the model to seamlessly generalise to any number of visual inputs, regardless of how many are used during training. In particular, we use only up to 5 images per sequence when training on our interleaved datasets, yet our model is able to benefit from sequences of up to 32 pairs (or “shots”) of images/videos and corresponding texts during evaluation.
 - **Flamingo can also process Videos**: **The Perceiver Resampler** module maps a variable size grid of **spatio-temporal visual features output** by the Vision Encoder to a fixed number of output tokens, independently from the input image resolution or the number of input video frames. This transformer has a set of **learned latent vectors** as **queries** and the **keys** and **values** are a concatenation of the spatio-temporal visual features with the learned latent vectors.

**When adding tokens to the data**:
Given text interleaved with images/videos. We first process the text by inserting `<image>` tags at the locations of the visual data in the text as well as **special tokens** such as : `<BOS>` for "beginning of sequence" or <EOC> for "end of chunk".
Images are processed independently by the **Vision Encoder** and **Perceiver Resampler** to extract visual tokens
**Processing procedure**: At a given text token, the model only cross-attends to the visual tokens corresponding to the last preceding image/video


## **Potential Architecture**
![image](https://github.com/user-attachments/assets/8122bdd2-67be-46f4-9c94-c97d4d88b53b)
![image](https://github.com/user-attachments/assets/273d49fe-05c7-4789-b114-f1698e42ec8a)
![image](https://github.com/user-attachments/assets/c293558a-6ead-4e20-a3a4-e6db71ec4f01)



## **Questions**
 - Multi-class classification ?--
 - How to split the data into training, validation and test?--
## **Important Questions**
 - What should we do with the text in the frames, should we only remove subtitles which distort the images? (because ViT processes them better?)
 - Should we use Fine-Tuned Bert instead of DistillBert?
 - Should we use sentence transformer?
 - How to make the transcription starts when the conversation actually starts (manuallly)?
 - Should we fine-tune the OCR with memes?


## **TO-DO**
 - Add the inference part for TrOCR fine-tuning (Mert) --
 - BERT fine-tuning with hatespeech tweets (Amene) --
 - Annotation of HateMM (Noah)--
 - ViT architecture analysis (Noah)

### **Optional TO-DO**
 - Go deep in the past for literature review --
 - Extract videos using vide_download.py for MultiHateClip --



