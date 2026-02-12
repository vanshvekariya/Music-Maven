# Music Maven: MP2.G3 – Advanced Beat Tracking & Temporal Query Analysis

**Project Design Specification** **Course:** CSC475 / CSC575 – Music Information Retrieval  
**Date:** February 2026

## 1. Project Overview

As a core subsystem of the **Music Maven** conversational AI ecosystem, **MP2.G3** focuses on the temporal analysis of audio signals. While Large Language Models (LLMs) excel at processing text, they lack the intrinsic ability to "hear" the rhythm, tempo, or meter of a raw audio file. Our module bridges this gap by implementing a robust signal processing and deep learning pipeline that extracts temporal features (BPM, beat positions, downbeats, and meter) from the **Music4All** dataset.

The system is designed to enable the Music Maven chatbot to answer queries such as:
* "Is this song fast enough for a high-intensity workout?"
* "Find me a song with a similar groove to 'Billie Jean' but slower."
* "What is the time signature of this track?"

This module will operate as a microservice within the larger Music Maven architecture, exposing temporal analysis endpoints to the orchestration layer.

---

## 2. Methodology & Architecture

Our approach combines traditional Digital Signal Processing (DSP) with modern Deep Learning (DL) techniques to ensure high accuracy across diverse genres.

### 2.1 Core Components
1. **Signal Preprocessing:** Normalization and spectrogram generation using `librosa` and `essentia`.
2. **Rhythm Analysis Engine:**
    * **Deterministic:** Onset detection functions and dynamic programming for standard beat tracking.
    * **Probabilistic (Advanced):** Recurrent Neural Networks (RNNs) or Temporal Convolutional Networks (TCNs) for extracting beat activation functions in complex audio.
3. **Semantic Mapping Layer:** A translation layer that converts numerical data (e.g., "128 BPM") into semantic descriptors (e.g., "Energetic", "Driving") for LLM consumption.

### 2.2 Integration Strategy
To align with **MP2.G1 (Architecture)**, this module will be deployed as a **FastAPI** service. This ensures modularity, allowing the central chatbot agent to query our service asynchronously without blocking the conversation flow.

---

## 3. Milestone Breakdown

### **BASIC** – Foundation & Standard Extraction
*Goal: Establish the processing pipeline and implement standard DSP-based beat tracking.*

* **Setup:** Initialize repository structure and dependency management (Poetry/pip).
* **Data Pipeline:** Implement audio loading from the Music4All dataset (previews/clips).
* **Baseline Algo:** Implement standard dynamic programming beat tracking (Ellis/Librosa method).
* **Basic API:** Create a CLI tool to input an audio file and output a raw BPM integer.

### **EXPECTED** – Downbeat Tracking & Semantic Integration
*Goal: Enhance accuracy with machine learning and integrate with the chatbot's query logic.*

* **ML-Based Tracking:** Replace/Augment DSP methods with RNN-based beat tracking (using `madmom` or similar pre-trained models).
* **Downbeat Detection:** Distinguish between the "beat" and the "downbeat" (the 1 of the measure) to infer bar lines.
* **Semantic API:** Expose an API endpoint that returns a JSON object containing `bpm`, `confidence`, and `tempo_class` (e.g., "Adagio", "Allegro").
* **Evaluation:** Run benchmarks against a labeled subset of Music4All.

### **ADVANCED** – Deep Learning & Complex Meter Analysis
*Goal: Implement state-of-the-art Deep Learning models for complex rhythm analysis and high-level descriptors.*

* **TCN/Transformer Model:** Implement a Temporal Convolutional Network or Transformer-based model for beat tracking in difficult genres (Jazz, Classical).
* **Meter Classification:** Automatically classify time signatures (4/4, 3/4, 6/8).
* **LLM "Vibe" Mapping:** Fine-tune the semantic mapping to align with user intent (e.g., mapping "aggressive" queries to high-BPM + high-onset density tracks).
* **Micro-timing Analysis:** Analyze "groove" (swing ratios) to differentiate between straight and swung rhythms.

---

## 4. Individual Contributions

### **Bharath Irukulapati**

**Objective: Deep Learning Architecture & System Integration**
* **PI.1 (Basic):** Design and implement the FastAPI microservice architecture with Pydantic schemas to serve beat tracking predictions to the central Music Maven chatbot.
* **PI.2 (Basic):** Set up the Deep Learning pipeline (PyTorch/TensorFlow) to convert Music4All audio into Mel-spectrograms suitable for neural network input.
* **PI.3 (Expected):** Implement a Recurrent Neural Network (RNN) or verify a pre-trained Transformer model to extract precise downbeat locations (identifying the "1" of the bar).
* **PI.4 (Expected):** Develop a "Semantic Mapper" that translates numerical outputs (BPM, Meter) into conversational contexts (e.g., mapping >170 BPM to "High Intensity") for the LLM.
* **PI.5 (Advanced):** Implement a "Groove Analysis" module that calculates swing ratios (micro-timing deviations) to distinguish between "straight" and "swung" rhythms, optimizing inference latency for near real-time performance.

---

### **Robert Widjaja**

**Objective: Establish Baseline Tempo Estimation Pipeline**
* **PI.1 (Basic):** Write scripts to batch-process audio files from the Music4All dataset, handling format conversion and sampling rate normalization.
* **PI.2 (Basic):** Implement standard onset strength envelope extraction using `librosa` to visualize rhythm features.
* **PI.3 (Expected):** Develop a comparative analysis script that benchmarks different standard algorithms (e.g., autocorrelation vs. comb filters) for global BPM estimation.
* **PI.4 (Expected):** Implement a caching mechanism to store computed BPMs for the dataset to avoid re-processing.
* **PI.5 (Advanced):** Create a "confidence score" metric that flags tracks where the tempo is ambiguous or fluctuating.

---

### **Apoorva Chadda**

**Objective: Build a music feature dataset, cluster songs, and identify originals from short audio or lyrics snippets.**

* **PI.1 (Advanced):** Build a dataset contaning below features against each audio file:

	Timbre (how the song sounds)
	These are the most important for clustering.
	- MFCCs (mean + variance)
	- Spectral centroid
	- Spectral bandwidth
	- Spectral contrast
	- Spectral rolloff
	- Zero-crossing rate
	- RMS energy
	These describe brightness, warmth, sharpness, noisiness, etc.

	Rhythm (how the song moves)
	Useful for grouping songs with similar groove.
	- Tempo (BPM)
	- Beat histogram
	- Inter-beat intervals (cadence)
	- Onset strength statistics

	Harmony (what chords/notes dominate)
	Great for genre clustering.
	- Chroma features (mean + variance)
	- Tonnetz (tonal centroid features)

	Energy & Dynamics
	Captures loudness patterns.
	- RMS energy (mean, variance)
	- Mel-spectrogram energy distribution

* **PI.2 (Advanced):** Using the above dataset, we will implement unsupervised machine learning through DBSCAN to cluster similar types of songs in the same groupings.

	-Identify songs in the same clusters by breaking them down into themes of genre, energy, and cadence - In case compute complexity increases, instead of using pandas, we will be able to using Spark Python.

* **PI.3 (Advanced):**  User-generated music often modifies audio (tempo, beats, remix, lo-fi) while keeping lyrics unchanged. Identify the original song from short audio segments or lyrics snippets, even under transformations. Support robust matching for partial or distorted inputs. Inspired by Shazam, SoundHound, and YouTube Content ID for copyright enforcement.

	- Use segment-level audio similarity and lyrics-based text similarity (cosine & Jaccard) for reliable identification.


---

## 5. Tools & Technologies

* **Languages:** Python 3.10+
* **Deep Learning:** PyTorch, TensorFlow
* **Signal Processing:** `librosa`, `madmom`, `essentia`
* **API Framework:** FastAPI, Pydantic
* **Data Handling:** `pandas`, `numpy`
* **Dataset:** Music4All (Audio Previews & Metadata)

---

## 6. Bibliography

1. **Ellis, D. P. (2007).** "Beat tracking by dynamic programming." *Journal of New Music Research*, 36(1), 51-60.
2. **Böck, S., & Schedl, M. (2011).** "Enhanced beat tracking with context-aware neural networks." *Proceedings of the 14th International Conference on Digital Audio Effects (DAFx-11)*.
3. **Davies, M. E., & Plumbley, M. D. (2007).** "Context-dependent beat tracking of musical audio." *IEEE Transactions on Audio, Speech, and Language Processing*, 15(3), 1009-1020.
4. **Schreiber, H., & Müller, M. (2018).** "A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network." *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR)*.
5. **Böck, S., Krebs, F., & Widmer, G. (2019).** "Joint Beat and Downbeat Tracking with Recurrent Neural Networks." *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*.
6. **Gouyon, F., et al. (2006).** "An experimental comparison of audio tempo induction algorithms." *IEEE Transactions on Audio, Speech, and Language Processing*, 14(5), 1832-1844.
7. **Krebs, F., Böck, S., & Widmer, G. (2015).** "An Efficient State Space Model for Joint Tempo and Meter Tracking." *Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR)*.
8. **Oramas, S., et al. (2017).** "Multi-label Music Genre Classification from Audio, Text, and Images Using Deep Features." *Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)*.
9. **Music4All Dataset Team.** (2020). "Music4All: A New Music Database." *Proceedings of the 2020 International Conference on Multimedia Retrieval*.
10. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems (NIPS)*.
11. **Dixon, S. (2001).** "Automatic extraction of tempo and beat from expressive performances." *Journal of New Music Research*, 30(1), 39-58.
12. **Hung, H. T., et al. (2021).** "Modeling Beat and Downbeat Tracking with a Temporal Convolutional Network." *Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR)*.
13. **Agostinelli, A., et al. (2023).** "MusicLM: Generating Music From Text." *arXiv preprint arXiv:2301.11325*.
14. **Gemmeke, J. F., et al. (2017).** "Audio Set: An ontology and human-labeled dataset for audio events." *IEEE ICASSP*.
15. **McFee, B., et al. (2015).** "librosa: Audio and music signal analysis in python." *Proceedings of the 14th python in science conference*.
16. **Klapuri, A. P., et al. (2006).** "Analysis of the meter of acoustic musical signals." *IEEE Transactions on Audio, Speech, and Language Processing*, 14(1), 342-355.
17. **Shao, Y., et al. (2020).** "JukeBox: A Generative Model for Music." *arXiv preprint*.
18. **Gkiokas, A., et al. (2012).** "Music Tempo Estimation and Beat Tracking." *Springer Handbook of Systematic Musicology*.
