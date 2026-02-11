# Music-Maven – MP2.G3 Beat Tracking & Tempo Queries

CSC475 / CSC575 Mega Project Specification

## Overview

Music-Maven is a music-aware conversational AI system capable of analyzing audio recordings and responding intelligently to music-related queries. The system integrates Large Language Models (LLMs) with audio signal processing techniques to provide domain-specific music understanding.

This sub-project (MP2.G3) focuses on integrating beat tracking and tempo-related functionality into the system.

---

## MP2.G3 – Beat Tracking & Tempo Queries

The goal of this component is to enable the chatbot to analyze real audio signals and respond to tempo-related questions. The system will extract tempo (BPM) and beat information from audio files and convert those numerical results into meaningful conversational responses.

The implementation will rely on existing audio processing libraries (e.g., librosa), with emphasis on integration, evaluation, and explanation rather than developing new signal processing algorithms.

---

# Individual Contribution

## Robert Widjaja
## Apoorva Chadda

### Objective: Implement and evaluate a functional tempo estimation and beat tracking module integrated into the Music-Maven chatbot.

PI1 (basic): Load and preprocess audio files from the Music4All dataset for analysis.

PI2 (basic): Developing new features by extracting tempo (BPM) using a beat-tracking algorithm and return numerical tempo values.

PI3 (expected): Extract beat timestamps and generate natural-language responses for tempo-related user queries (e.g., “Is this song fast or slow?”).

PI4 (expected): Demonstrate functionality on at least 5 different audio samples and report tempo estimation results.

PI5 (advanced): Analyze tempo estimation behavior across a subset of the dataset (e.g., summarize BPM distribution or compare tempo outputs for different audio segments of the same track).

PI6 (advanced): Identify similar segments/genres/profile songs using unsupervised machine learning. 

---

## Expected Deliverables

- Working tempo estimation and beat extraction module
- Conversational handling of tempo-related queries
- Demonstration using Music4All samples
- Basic evaluation of system behavior

---

## Limitations

- Tempo estimation is approximate and dependent on audio characteristics.
- Beat tracking accuracy may vary across musical genres.
- No custom model training is performed in this module.

---