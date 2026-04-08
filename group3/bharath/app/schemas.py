"""
Pydantic schemas for the Beat Tracking Microservice.
Defines request/response models for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ── Enums ──

class TempoClass(str, Enum):
    """Musical tempo classifications based on Italian tempo markings."""
    LARGO = "Largo"            # 40-60 BPM
    ADAGIO = "Adagio"          # 60-80 BPM
    ANDANTE = "Andante"        # 80-100 BPM
    MODERATO = "Moderato"      # 100-120 BPM
    ALLEGRO = "Allegro"        # 120-156 BPM
    VIVACE = "Vivace"          # 156-176 BPM
    PRESTO = "Presto"          # 176-200 BPM


class EnergyLevel(str, Enum):
    """Semantic energy descriptors for LLM consumption."""
    LOW = "Low Intensity"
    MODERATE = "Moderate Intensity"
    HIGH = "High Intensity"
    VERY_HIGH = "Very High Intensity"


class MeterType(str, Enum):
    """Common musical time signatures."""
    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"
    SIX_EIGHT = "6/8"
    UNKNOWN = "Unknown"


# ── Request Models ──

class BeatTrackingRequest(BaseModel):
    """Request to analyze an audio file for beat/tempo information."""
    file_path: str = Field(
        ...,
        description="Path to the audio file to analyze",
        examples=["data/audio/12345.mp3"]
    )
    include_beats: bool = Field(
        default=True,
        description="Whether to return individual beat timestamps"
    )
    include_downbeats: bool = Field(
        default=True,
        description="Whether to return downbeat (bar start) timestamps"
    )


class SemanticQueryRequest(BaseModel):
    """Request to translate tempo/rhythm data into semantic descriptors."""
    bpm: float = Field(
        ..., gt=0, le=300,
        description="Beats per minute",
        examples=[128.0]
    )
    meter: Optional[str] = Field(
        default=None,
        description="Time signature string",
        examples=["4/4"]
    )
    onset_density: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Normalized onset density (0=sparse, 1=dense)"
    )


class BatchAnalysisRequest(BaseModel):
    """Request to analyze multiple audio files."""
    file_paths: List[str] = Field(
        ..., min_length=1,
        description="List of audio file paths"
    )


# ── Response Models ──

class BeatTrackingResponse(BaseModel):
    """Full beat tracking analysis result."""
    file_path: str
    bpm: float = Field(..., description="Estimated tempo in BPM")
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Confidence score of the BPM estimate"
    )
    beat_times: Optional[List[float]] = Field(
        default=None,
        description="Timestamps (seconds) of detected beats"
    )
    downbeat_times: Optional[List[float]] = Field(
        default=None,
        description="Timestamps (seconds) of detected downbeats"
    )
    meter: MeterType = Field(
        default=MeterType.UNKNOWN,
        description="Estimated time signature"
    )
    tempo_class: TempoClass = Field(
        ..., description="Italian tempo classification"
    )
    energy_level: EnergyLevel = Field(
        ..., description="Semantic energy descriptor for LLM"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of the audio file"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str = "beat-tracking"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
