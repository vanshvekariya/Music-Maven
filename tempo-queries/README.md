# MP2.G3 – Beat Tracking & Tempo-Related Queries

CSC475 / CSC575 Mega Project Specification  
Music-Maven Sub-Project

## Overview

This sub-project focuses on integrating beat tracking and tempo analysis into the Music-Maven chatbot system. The goal is to enable the chatbot to analyze real audio recordings and respond intelligently to tempo-related music queries.

Rather than developing new signal processing algorithms from scratch, this project integrates existing beat-tracking libraries and exposes their outputs through a conversational interface.

---

## Motivation

Most general-purpose chatbots can discuss music conceptually but cannot analyze actual audio signals. This sub-project addresses that limitation by enabling:

- Automatic tempo (BPM) estimation
- Beat detection from audio files
- Conversational explanations of rhythmic characteristics
- Tempo-based reasoning (e.g., fast vs slow classification)

This module contributes real audio intelligence to the overall Music-Maven system.

---

## Objectives (Individual Contribution)

1. Implement tempo (BPM) estimation from audio files.
2. Extract beat timestamps from tracks.
3. Support tempo-related natural language queries.
4. Provide interpretable explanations of tempo results.
5. Integrate tempo functionality with the main chatbot architecture.

---

## Key Performance Indicators (KPIs)

The project will be considered successful if the following are achieved:

1. The system correctly estimates BPM for sample tracks from the Music4All dataset.
2. Beat timestamps are successfully extracted for analyzed tracks.
3. The chatbot correctly answers at least the following query types:
   - “What is the tempo of this song?”
   - “Is this song fast or slow?”
   - “Where are the beats?”
   - “Which of these two songs is faster?”
4. The tempo module integrates with the main Music-Maven routing system.
5. A successful demonstration is performed using at least 5 real audio samples.

---

## Technical Approach

### Audio Analysis

Tempo and beat detection will be implemented using existing Python libraries such as:

- librosa (for beat tracking and tempo estimation)

The system will:

- Load an audio file
- Estimate tempo in beats per minute (BPM)
- Extract beat frame positions and convert them to timestamps

### Tempo Categorization

Estimated BPM values will be categorized as:

- Below 80 BPM → Slow
- 80–110 BPM → Moderate
- 110–140 BPM → Fast
- Above 140 BPM → Very Fast

These categories will be used to generate conversational explanations.

### Query Handling

Tempo-related keywords will trigger the beat-tracking module. Examples include:

- tempo
- BPM
- fast
- slow
- beats
- rhythm

The module will return structured tempo data which is then converted into a natural-language response.

---

## Dataset

The Music4All dataset will be used for testing and evaluation. Sampled tracks will be analyzed to:

- Validate BPM estimation
- Demonstrate tempo distribution across songs
- Provide examples for conversational queries

Full dataset preprocessing is not required; only selected samples will be used for demonstration and validation.

---

## Expected Deliverables

- Working tempo estimation module
- Beat extraction functionality
- Conversational tempo query handling
- Integration with Music-Maven chatbot
- Demonstration using Music4All samples

---

## Limitations

- Tempo estimation is approximate and may vary across genres.
- Beat detection accuracy depends on audio quality.
- No custom model training is performed in this module.

---

## Future Extensions

- Beat visualization
- Tempo confidence scoring
- Rhythmic pattern analysis
- Danceability estimation

---

## Contributor

Robert Widjaja  
MP2.G3 – Beat Tracking & Tempo-Related Queries  
CSC475 / CSC575  
University of Victoria
