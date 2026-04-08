"""
Music Maven MP2.G3 – Beat Tracking Microservice
FastAPI application exposing temporal analysis endpoints.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time

from app.schemas import (
    BeatTrackingRequest,
    BeatTrackingResponse,
    SemanticQueryRequest,
    HealthResponse,
    ErrorResponse,
    TempoClass,
    EnergyLevel,
    MeterType,
)

# ── App Configuration ──

app = FastAPI(
    title="Music Maven – Beat Tracking Service",
    description=(
        "MP2.G3 microservice for temporal analysis of audio signals. "
        "Provides BPM estimation, beat/downbeat detection, meter classification, "
        "and semantic tempo descriptors for LLM integration."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper Functions ──

def classify_tempo(bpm: float) -> TempoClass:
    """Map a BPM value to its Italian tempo classification."""
    if bpm < 60:
        return TempoClass.LARGO
    elif bpm < 80:
        return TempoClass.ADAGIO
    elif bpm < 100:
        return TempoClass.ANDANTE
    elif bpm < 120:
        return TempoClass.MODERATO
    elif bpm < 156:
        return TempoClass.ALLEGRO
    elif bpm < 176:
        return TempoClass.VIVACE
    else:
        return TempoClass.PRESTO


def classify_energy(bpm: float, onset_density: Optional[float] = None) -> EnergyLevel:
    """Map BPM and onset density to a semantic energy descriptor."""
    # If onset density is provided, use a combined heuristic
    if onset_density is not None:
        score = (bpm / 200.0) * 0.6 + onset_density * 0.4
    else:
        score = bpm / 200.0

    if score < 0.35:
        return EnergyLevel.LOW
    elif score < 0.55:
        return EnergyLevel.MODERATE
    elif score < 0.75:
        return EnergyLevel.HIGH
    else:
        return EnergyLevel.VERY_HIGH


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the service is running."""
    return HealthResponse()


@app.post(
    "/analyze",
    response_model=BeatTrackingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Analysis"],
)
async def analyze_audio(request: BeatTrackingRequest):
    """
    Analyze an audio file for beat tracking information.

    Returns BPM, beat/downbeat timestamps, meter, and semantic descriptors.
    This endpoint will be backed by the full DL pipeline (PI.2/PI.3).
    Currently returns a placeholder to validate the API contract.
    """
    # ── Placeholder: will be replaced by real analysis in PI.2 & PI.3 ──
    # For now, return a mock response to prove the API works end-to-end
    import random
    mock_bpm = round(random.uniform(70, 180), 1)

    return BeatTrackingResponse(
        file_path=request.file_path,
        bpm=mock_bpm,
        confidence=0.85,
        beat_times=[0.5, 1.0, 1.5, 2.0] if request.include_beats else None,
        downbeat_times=[0.5, 2.0] if request.include_downbeats else None,
        meter=MeterType.FOUR_FOUR,
        tempo_class=classify_tempo(mock_bpm),
        energy_level=classify_energy(mock_bpm),
        duration_seconds=210.0,
    )


@app.post(
    "/semantic",
    tags=["Semantic"],
    summary="Translate numerical tempo data into LLM-friendly descriptors",
)
async def get_semantic_descriptors(request: SemanticQueryRequest):
    """
    Given raw BPM and optional meter/onset data, return semantic labels
    suitable for injection into an LLM prompt.
    """
    tempo_class = classify_tempo(request.bpm)
    energy = classify_energy(request.bpm, request.onset_density)

    return {
        "bpm": request.bpm,
        "tempo_class": tempo_class.value,
        "energy_level": energy.value,
        "meter": request.meter or "Unknown",
        "llm_context": (
            f"This track has a tempo of {request.bpm:.0f} BPM, "
            f"classified as {tempo_class.value}. "
            f"The energy level is {energy.value}."
        ),
    }


@app.get("/", tags=["System"])
async def root():
    """Service info."""
    return {
        "service": "Music Maven – Beat Tracking (MP2.G3)",
        "version": "0.1.0",
        "endpoints": ["/health", "/analyze", "/semantic", "/docs"],
    }
