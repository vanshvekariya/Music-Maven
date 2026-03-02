#  Music Maven  
**University of Victoria**  
**MP2 · Group 6**

---

## Project Overview

**Music Maven** is a music-specific AI chatbot capable of:

- Playing music  
- Analyzing raw audio  
- Understanding structured music queries  
- Returning intelligent, structured responses  

The system combines:

- Large Language Models (LLMs)  
- Audio representation learning  
- Genre classification  
- Similarity retrieval  
- Multi-label auto-tagging  

The goal is to create an **audio-aware conversational system** that can analyze a song and incorporate that analysis into meaningful dialogue responses.

---

# Audio Intelligence Module

This sub-module powers Music Maven’s ability to analyze raw audio files.

It performs:

- Genre classification  
- Similarity retrieval  
- Multi-label auto-tagging  

All outputs are returned as structured JSON compatible with the chatbot API.

# Current Status of Implementation
A baseline system has already been implemented, including audio preprocessing, mel-spectrogram and MFCC feature extraction using Librosa, and supervised genre classification using SVM, Random Forest, and a CNN model on the GTZAN dataset.
The remaining components — contrastive representation learning, embedding evaluation, similarity retrieval, and multi-label auto-tagging — will be implemented in subsequent milestones. This proposal therefore describes both the completed baseline and the planned research extensions.


# Methodological Justification 
This project follows established practices in Music Information Retrieval (MIR).
Raw waveforms are not directly suitable for learning because music perception depends on frequency structure. Therefore, audio is transformed into mel-spectrogram representations, which approximate human auditory perception and preserve harmonic and timbral information (Tzanetakis & Cook, 2002).
Supervised learning alone is limited because music labels are often incomplete. To address this, we use contrastive representation learning (SimCLR), which learns semantic similarity by comparing augmented versions of the same signal (Chen et al., 2020). Contrastive learning has been shown effective for large-scale audio representation learning (Kong et al., 2020).
Embedding quality is validated using k-nearest neighbor (k-NN) accuracy and UMAP/t-SNE visualization. If meaningful, songs of similar genres cluster together.
Finally, auto-tagging is treated as a multi-label classification problem since songs can contain multiple attributes simultaneously. A sigmoid output is used instead of softmax to allow independent tag prediction (Khosla et al., 2020).

---

# System Architecture

## 1. Audio Feature Extraction

**Preprocessing**
- Resample to 22,050 Hz  
- Convert to mono  
- Amplitude normalization  
- Fixed-length segmentation  

**Extracted Features**
- Mel-spectrograms  
- MFCC (40 coefficients)  
- Spectral centroid  
- Chroma  
- Zero-crossing rate  
- Tempo  

---

## 2. Contrastive Representation Learning

- CNN encoder trained using **SimCLR**
- NT-Xent loss

**Augmentations**
- Pitch shift  
- Time stretch  
- Additive noise  
- Time/frequency masking  
- Random cropping  

**Evaluation**
- k-NN classification accuracy  
- Cosine similarity retrieval  
- UMAP / t-SNE clustering  

---

## 3️. Genre Classification

**Models**
- Logistic Regression  
- SVM (RBF)  
- Random Forest  
- CNN on mel-spectrograms  

**Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- **Macro F1 (Primary)**  
- Confusion Matrix  

**Comparison**
- MFCC features vs contrastive embeddings  
- Ablation: with vs without contrastive pre-training  

---

## 4️. Multi-Label Auto-Tagging

Formulated as a **multi-label classification** problem.

**Tag Categories**
- Genre  
- Mood  
- Instrumentation  
- Tempo  

**Model**
- Sigmoid-output MLP trained on frozen embeddings  

**Metrics**
- Micro F1  
- Macro F1  
- Hamming Loss  
- ROC-AUC  
- Precision@K  

---

## 5️. API Integration

Deployment via **FastAPI** microservice.

### Example Request

```json
{
  "type": "audio_tag",
  "audio_url": "...",
  "top_k_genres": 3,
  "include_moods": true,
  "include_instruments": true
}
```

# Evaluation Strategy

## Representation Learning
- kNN accuracy  
- Clustering visualization (UMAP / t-SNE)  
- Cosine similarity ranking  

## Genre Classification (Multi-Class)
- Accuracy  
- Precision  
- Recall  
- Macro F1  
- Confusion Matrix  

## Auto-Tagging (Multi-Label)
- Micro F1  
- Macro F1  
- Hamming Loss  
- ROC-AUC  

---

# ⚠ Risks & Limitations

- Noisy labels (GTZAN dataset)  
- Genre ambiguity (e.g., Rock vs Alternative)  
- Class imbalance  
- Short segment limitations  
- Overfitting risk  
- Cross-dataset generalization concerns  

---

# Milestones

| Period | Objective |
|--------|-----------|
| Feb 15 – Feb 28 | Feature extraction & baseline models |
| Mar 1 – Mar 31 | Contrastive encoder & classification |
| Mar 1 – Mar 31 | Auto-tagging & API |

---

# Individual Responsibilities & Performance Indicators

---

## Aman Monga – Contrastive Learning & Embeddings

### Objective 1: Contrastive Representation Learning

PI.1 (basic): Implement audio augmentations (pitch shift, time stretch, masking, noise).  
PI.2 (expected): Train SimCLR encoder using NT-Xent loss.  
PI.3 (expected): Export learned embeddings for all tracks.  
PI.4 (expected): Evaluate embeddings using k-NN classification.  
PI.5 (advanced): Generate UMAP/t-SNE visualizations showing genre clustering.  

### Objective 2: Embedding Validation & Analysis

PI.1 (basic): Compare k-NN accuracy vs MFCC baseline.  
PI.2 (expected): Evaluate cosine similarity retrieval quality.  
PI.3 (expected): Perform ablation (with vs without pretraining).  
PI.4 (advanced): Test embedding transferability on secondary dataset.  

### Final Deliverables
- Trained encoder (.pt file)  
- Exported embedding dataset  
- Visualization plots  
- Evaluation report  

---

## Newsha Bahardoost – Genre Classification

### Objective 1: Supervised Genre Classification

PI.1 (basic): Train Logistic Regression baseline on MFCC features.  
PI.2 (expected): Train SVM and Random Forest models.  
PI.3 (expected): Train CNN on mel-spectrogram inputs.  
PI.4 (expected): Evaluate models (Accuracy, Macro F1, Confusion Matrix).  
PI.5 (advanced): Hyperparameter tuning and model comparison.  

### Objective 2: Model Comparison & Analysis

PI.1 (basic): Compare MFCC features vs contrastive embeddings.  
PI.2 (expected): Conduct misclassification analysis.  
PI.3 (expected): Select best-performing classifier.  
PI.4 (advanced): Perform ablation study on pretraining impact.  

### Final Deliverables
- Best-performing classifier  
- Evaluation metrics report  
- Confusion matrix visualizations  

---

## Kevin Nguyen – Auto-Tagging & API Integration

### Objective 1: Multi-Label Auto-Tagging

PI.1 (basic): Define structured tag vocabulary (genre, mood, instruments, tempo).  
PI.2 (expected): Train sigmoid-based MLP on frozen embeddings.  
PI.3 (expected): Tune classification thresholds.  
PI.4 (expected): Evaluate with Micro F1, Macro F1, ROC-AUC.  
PI.5 (advanced): Optimize Precision@K and reduce Hamming Loss.  

### Objective 2: API Deployment & Integration

PI.1 (basic): Build FastAPI endpoint for audio tagging.  
PI.2 (expected): Return structured JSON outputs.  
PI.3 (expected): Integrate classifier with chatbot system.  
PI.4 (advanced): Perform end-to-end testing with live queries.  

### Final Deliverables
- Trained auto-tagger model  
- Functional FastAPI microservice  
- Structured JSON response format  
- Deployment documentation  

---

# Tools & Libraries

- PyTorch  
- Librosa / Torchaudio  
- Scikit-learn  
- UMAP / Matplotlib  
- FastAPI  

---

# Datasets

- GTZAN Genre Collection  
- Music4All  

---

# Key References

- Tzanetakis & Cook (2002) — Genre Classification  
- Chen et al. (2020) — SimCLR  
- Kong et al. (2020) — PANNs  
- Khosla et al. (2020) — Supervised Contrastive Learning  
- Humphrey et al. (2013) — Deep Learning for MIR  
- Choi et al. (2016) — Deep Music Tagging  
- Hershey et al. (2017) — CNN Audio Classification 

### Genre Classification
Tzanetakis, G., & Cook, P. (2002). Musical Genre Classification of Audio Signals. IEEE Transactions on Speech and Audio Processing.
https://doi.org/10.1109/TSA.2002.800560

### Contrastive Learning 
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML 2020.
View paper (arXiv PDF) 

### Audio Representation Learning 
Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM TASLP.
https://arxiv.org/abs/1912.10211

### Supervised Contrastive Learning
Khosla, P., Teterwak, P., Wang, C., et al. (2020). Supervised Contrastive Learning. NeurIPS.
https://arxiv.org/abs/2004.11362

### Deep Learning for Music
Humphrey, E. J., Bello, J. P., & LeCun, Y. (2013). Feature Learning and Deep Architectures: New Directions for Music Informatics.
https://arxiv.org/abs/1306.6458

### Music Auto-Tagging 
Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2016). Automatic Tagging Using Deep Convolutional Neural Networks. ISMIR.
https://arxiv.org/abs/1606.00298

### Audio CNN Architectures
Hershey, S., Chaudhuri, S., Ellis, D., et al. (2017). CNN Architectures for Large-Scale Audio Classification. ICASSP.
https://arxiv.org/abs/1609.09430

### Evaluation & Surveys 
Fu, Z., Lu, G., Ting, K. M., & Zhang, D. (2011). A Survey of Audio-Based Music Classification and Annotation. IEEE Transactions on Multimedia.
https://ieeexplore.ieee.org/document/5694077

---

# Final Goal

Deliver a fully integrated **music-aware chatbot** that can:

- Understand music queries  
- Analyze uploaded audio  
- Retrieve similar songs  
- Auto-tag tracks  
- Provide structured intelligent responses  

All components will be version-controlled, documented, and pushed to GitHub with clear setup and usage instructions for teammates and API consumers.