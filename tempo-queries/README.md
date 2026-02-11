# 🎵 Music-Maven — Tempo & Beat Queries (MP2.G3) (ONGOING)

## Overview

This component of Music-Maven implements tempo- and beat-related music queries as part of a music-aware chatbot. The goal is to integrate existing beat-tracking tools and expose their results through natural-language responses, rather than training new audio models. This work corresponds to MP2.G3: Integrating beat trackers and tempo-related queries.

## Objectives

- Estimate tempo (BPM) from audio files
- Detect beat locations
- Classify tempo as slow, moderate, fast, or very fast
- Answer tempo-related natural language queries
- Provide explainable, demo-ready outputs

## Design Philosophy

- Use off-the-shelf audio analysis libraries
- Avoid custom DSP or model training
- Emphasize system integration and interpretation
- Keep scope small, robust, and reliable

## System Architecture

User Query → Intent Interpretation → Audio Analysis (beat tracker) → Tempo and Beat Extraction → Natural Language Response

## Supported Query Types

- What is the tempo (BPM) of this song?
- Is this song fast or slow?
- Where are the beats in this track?
- Is this song suitable for dancing?
- Compare the tempo of two songs

## Tempo Categorization

Tempo is interpreted using common BPM ranges:

- Below 80 BPM: Slow
- 80–110 BPM: Moderate
- 110–140 BPM: Fast
- Above 140 BPM: Very Fast

## Dataset

This module is evaluated using the Music4All dataset. Sampled tracks are analyzed to validate BPM extraction, demonstrate tempo distributions, and showcase example query responses. Full dataset preprocessing is not required; only sampled audio is used for testing and demonstration purposes.

## Dependencies

The implementation relies on the following Python libraries:

- librosa
- numpy
- soundfile

These can be installed using pip.

## Example Interaction (May change)

User: Is this song fast or slow?
Bot:
Estimated tempo: 128 BPM
Tempo category: Fast
This tempo is energetic and commonly suitable for dancing.

## Limitations

- Tempo estimation is approximate and may vary depending on audio quality and musical style
- Beat tracking accuracy may differ across genres
- This module focuses on integration rather than state-of-the-art tempo estimation
