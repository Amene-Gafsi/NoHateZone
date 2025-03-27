# NoHateZone

### **Our idea**
Use re-annotated meme, texts(tweets), audio speech and video datasets to ‘beep’ out audio chunks and blur out frames in videos where hate speech occurs.
The selected audio chunks and frames are going to fed into a simple neural network whose objective is to classify the object as the source of the hate speech: Race, Ethnicity, Religion and other.

### **Introduction**
Social media comprises platforms that allow users to connect with others to expand their knowledge, have a laugh, and discuss issues with global impacts. For social media to serve these purposes, it is essential to foster a safe environment where people can share content and interact without offending, humiliating, or bullying others. According to statistics shared by the FBI’s Uniform Crime Reporting Program, hate crimes occur in three main contexts: gender/sexual orientation, religion, and race/ethnicity. Together, these constitute 96% of the hate crimes reported in the United States in 2024. While social media enables a broad spectrum of self-expression across various topics, hate speech is not defined by its subject matter alone. Rather, hate speech is characterized by language that is hateful, discriminatory, or incites harm (journals.sagepub.com). This distinction underscores that although hate speech may appear in diverse forms, its defining criteria set it apart from general self-expression.

### **Contribution to on-going research**
Researches in the field of hate speech classification focuses on the end problem of classifying a video as hateful as not hateful. As in article (3), the authors proposed a deep learning framework to classify videos as hateful or not hateful by combining state-of-the-art transformers with a fusion layer. Although it is a novel approach, it doesn’t generate any addiitonal insight on the reasoning behind the classification. Our deep learning framework attempts to explain the full methodology behind the classification by providing frame-level and chunk-level rationale for audio component of a video.


### **Potential data sources**
#### **Text (tweets)**
**HateXplain dataset**
  •	Contains word-level annotations: a word w is assigned a rationale of 1 if it contributes to the classification of the sentence as hate speech, and 0 otherwise
#### **Memes** 
Article (1) states that In order to capture temporal dynamics of videos in classification problems, it is crucial to consider the three main modalities which are text, image and audio. As memes inherently combine visual and textual elements, they closely mirror the multimodal structure of videos. Therefore memes become a go-to choice to augment video-based hate speech datasets as they remain signficantly smaller compared to datasets beloning to other modalities.

	Facebook Hateful Memes dataset(FHM)
•	Licence found
	Multimedia automatic misogyny Identification(MAMI)
	Re-annoted meme dataset
Videos
	MultiHateClip (MHC)
	
	HateMM(HMM)



Cross-Modal Transfer from Memes to Videos: Addressing Data Scarcity in Hateful Video Detection (1)
The vision modality requires different handling, as videos consist of multiple frames, while memes are represented by single images.
For video-based VLMs, such as LLaVA-NeXT-Video-7B, which can process multiple frames, we sample 16 frames per video, aligning with the model author’s recommendations and the average length of our video dataset
To adapt meme data for video-based VLMs, we simulate a video format by applying random image augmentations (e.g., rotation, cropping, etc.) to generate 16 frames from a single meme image. We employ Low-Rank Adaptation (LoRA) adapters to fine-tune the pre-trained VLM during.

Hate Speech recognition in Audio dataset (2)
In this study, we combine both approaches to classify entire audio clips as containing hate speech or normal speech, while also pinpointing the exact segments within the audio where hate speech occurs, using a 10-millisecond time grid.
Cascading and End-to-End (E2E)
•	pinpoint hate speech within specific time frames

HateMM: A Multi-Modal Dataset for Hate Video Classification (3)
BitChute- labels along with the frame spans 
image-based hate detection methods cannot be directly adapted
Combined BERT-VIT-MFCC with the help of fusion layer

#### **Potential Architecture**
![image](https://github.com/user-attachments/assets/6e7f02e1-9834-42b9-82de-e492e26076ea)


Questions
MultiHateClip is classifed by ‘Gender’ hate, is this a problem? —
How to make use of HateXplain dataset? –
Do we need a seperate model for multi-class classification (check HateMM)?--
Should we analyze the frames 
Manual label 


TO-DO

Go deep in the past for literature review—
Extract videos using vide_download.py for MultiHateClip--



