"""
PI.4 – Semantic Mapper
Translates numerical beat tracking outputs (BPM, meter, onset density)
into natural language descriptors suitable for LLM prompt injection.

This module bridges the gap between raw signal processing outputs and
the Music Maven chatbot's language model. Rather than passing "128 BPM"
to the LLM, we provide rich semantic context like:
    "This track has an Allegro tempo (128 BPM), High energy, in 4/4 time.
     It would suit a running playlist or an upbeat dance setting."

The mapping is based on established musicological conventions (Italian
tempo markings) combined with empirical energy/mood heuristics.

Usage:
    from app.semantic_mapper import SemanticMapper
    mapper = SemanticMapper()
    context = mapper.map(bpm=128, meter=4, onset_density=0.7)
    print(context.llm_prompt)  # Rich text for LLM injection

References:
    - Italian tempo markings: standard Western music terminology
    - Russell's circumplex model of affect (valence-arousal space)
    - Musicological tempo-mood associations (Gabrielsson & Lindström, 2001)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import json


# ══════════════════════════════════════════════════════════════════════
# Enums and Data Classes
# ══════════════════════════════════════════════════════════════════════

class TempoMarking(Enum):
    """Italian tempo markings with associated BPM ranges.

    These are standard musical terminology used worldwide.
    Each marking carries both a speed and an expressive connotation.
    """
    LARGO      = ("Largo",      40,  60,  "Very slow, broad, stately")
    ADAGIO     = ("Adagio",     60,  80,  "Slow, at ease, restful")
    ANDANTE    = ("Andante",    80,  100, "Walking pace, moderate, flowing")
    MODERATO   = ("Moderato",   100, 120, "Moderate speed, steady")
    ALLEGRO    = ("Allegro",    120, 156, "Fast, lively, bright")
    VIVACE     = ("Vivace",     156, 176, "Lively, brisk, vibrant")
    PRESTO     = ("Presto",     176, 210, "Very fast, urgent, exciting")
    PRESTISSIMO = ("Prestissimo", 210, 300, "Extremely fast, as fast as possible")

    def __init__(self, label: str, bpm_min: int, bpm_max: int, description: str):
        self.label = label
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.description = description


class EnergyLevel(Enum):
    """Energy descriptors based on combined BPM and onset density."""
    VERY_LOW  = ("Very Low",   "calm, ambient, meditative")
    LOW       = ("Low",        "relaxed, gentle, soothing")
    MODERATE  = ("Moderate",   "balanced, steady, neutral")
    HIGH      = ("High",       "energetic, driving, powerful")
    VERY_HIGH = ("Very High",  "intense, explosive, frenetic")

    def __init__(self, label: str, descriptors: str):
        self.label = label
        self.descriptors = descriptors


class MoodCategory(Enum):
    """High-level mood categories derived from tempo and energy."""
    SERENE     = ("Serene",     "meditation, sleep, yoga, ambient background")
    MELANCHOLY = ("Melancholy", "reflective listening, rainy day, journaling")
    CHILL      = ("Chill",      "studying, coffee shop, casual hangout")
    GROOVY     = ("Groovy",     "cooking, commuting, light exercise")
    UPBEAT     = ("Upbeat",     "running, dancing, party warmup")
    INTENSE    = ("Intense",    "HIIT workout, gaming, high-energy dance")
    FRANTIC    = ("Frantic",    "sprinting, extreme sports, mosh pit")

    def __init__(self, label: str, use_cases: str):
        self.label = label
        self.use_cases = use_cases


class MeterDescription(Enum):
    """Human-readable meter descriptions."""
    DUPLE   = (2, "Duple meter (2/4) – march-like, binary feel")
    TRIPLE  = (3, "Triple meter (3/4) – waltz-like, lilting feel")
    COMMON  = (4, "Common time (4/4) – standard, most popular meter")
    COMPOUND = (6, "Compound meter (6/8) – rolling, pastoral feel")

    def __init__(self, beats: int, description: str):
        self.beats = beats
        self.description = description


@dataclass
class SemanticContext:
    """Complete semantic analysis suitable for LLM consumption.

    This is the primary output of the SemanticMapper, containing
    all the information needed to enrich an LLM prompt with
    musical context about a track's temporal characteristics.
    """
    bpm: float
    tempo_marking: str
    tempo_description: str
    energy_level: str
    energy_descriptors: str
    mood: str
    mood_use_cases: str
    meter: int
    meter_description: str
    confidence: float
    llm_prompt: str  # The complete formatted context string
    tags: List[str] = field(default_factory=list)  # Searchable tags

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "bpm": self.bpm,
            "tempo_marking": self.tempo_marking,
            "tempo_description": self.tempo_description,
            "energy_level": self.energy_level,
            "energy_descriptors": self.energy_descriptors,
            "mood": self.mood,
            "mood_use_cases": self.mood_use_cases,
            "meter": self.meter,
            "meter_description": self.meter_description,
            "confidence": self.confidence,
            "llm_prompt": self.llm_prompt,
            "tags": self.tags,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ══════════════════════════════════════════════════════════════════════
# Semantic Mapper
# ══════════════════════════════════════════════════════════════════════

class SemanticMapper:
    """Maps numerical tempo/rhythm data to semantic descriptors for LLM integration.

    This is the core translation layer between the signal processing
    pipeline (PI.2, PI.3) and the Music Maven chatbot. It converts
    raw numbers into rich, contextual descriptions that help the LLM
    generate musically informed responses.

    The mapping uses three dimensions:
        1. Tempo marking (BPM → Italian classification)
        2. Energy level (BPM + onset density → intensity)
        3. Mood category (tempo + energy → use-case context)
    """

    # ── Keyword mappings for natural language queries ──
    # These map user "vibe" words to BPM ranges, enabling the chatbot
    # to handle queries like "find me something chill" or "I need
    # high-energy workout music"
    VIBE_TO_BPM: Dict[str, tuple] = {
        # Calm vibes
        "calm":        (40, 80),
        "peaceful":    (40, 70),
        "meditative":  (40, 65),
        "ambient":     (50, 80),
        "sleepy":      (40, 60),
        "relaxing":    (50, 80),
        # Moderate vibes
        "chill":       (70, 110),
        "mellow":      (70, 100),
        "groovy":      (90, 120),
        "smooth":      (80, 110),
        "laid-back":   (75, 105),
        # Upbeat vibes
        "upbeat":      (110, 140),
        "happy":       (110, 140),
        "energetic":   (120, 160),
        "danceable":   (115, 135),
        "funky":       (100, 130),
        # High energy vibes
        "intense":     (140, 180),
        "aggressive":  (150, 200),
        "workout":     (130, 170),
        "running":     (150, 180),
        "hype":        (140, 180),
        "fast":        (150, 210),
    }

    def classify_tempo(self, bpm: float) -> TempoMarking:
        """Map BPM to Italian tempo marking.

        Args:
            bpm: Beats per minute (0-300)

        Returns:
            Matching TempoMarking enum value.
        """
        for marking in TempoMarking:
            if marking.bpm_min <= bpm < marking.bpm_max:
                return marking
        # Edge cases
        if bpm < 40:
            return TempoMarking.LARGO
        return TempoMarking.PRESTISSIMO

    def classify_energy(
        self, bpm: float, onset_density: Optional[float] = None
    ) -> EnergyLevel:
        """Map BPM and onset density to an energy level.

        Uses a weighted combination of normalized BPM and onset density.
        BPM contributes 60% and onset density 40% to the final score.

        Args:
            bpm: Beats per minute.
            onset_density: Normalized onset density [0, 1]. Optional.

        Returns:
            EnergyLevel enum value.
        """
        # Normalize BPM to [0, 1] assuming 40-210 range
        bpm_norm = max(0, min(1, (bpm - 40) / (210 - 40)))

        if onset_density is not None:
            score = bpm_norm * 0.6 + onset_density * 0.4
        else:
            score = bpm_norm

        if score < 0.15:
            return EnergyLevel.VERY_LOW
        elif score < 0.35:
            return EnergyLevel.LOW
        elif score < 0.55:
            return EnergyLevel.MODERATE
        elif score < 0.75:
            return EnergyLevel.HIGH
        else:
            return EnergyLevel.VERY_HIGH

    def classify_mood(self, tempo: TempoMarking, energy: EnergyLevel) -> MoodCategory:
        """Derive mood from tempo and energy combination.

        Uses a simple lookup mapping common tempo-energy pairs to moods.

        Args:
            tempo: Italian tempo classification.
            energy: Energy level classification.

        Returns:
            MoodCategory enum value.
        """
        # Map (tempo_speed, energy_level) to mood
        tempo_speed = {
            TempoMarking.LARGO: "slow", TempoMarking.ADAGIO: "slow",
            TempoMarking.ANDANTE: "medium", TempoMarking.MODERATO: "medium",
            TempoMarking.ALLEGRO: "fast", TempoMarking.VIVACE: "fast",
            TempoMarking.PRESTO: "very_fast", TempoMarking.PRESTISSIMO: "very_fast",
        }

        energy_level = {
            EnergyLevel.VERY_LOW: "low", EnergyLevel.LOW: "low",
            EnergyLevel.MODERATE: "mid", EnergyLevel.HIGH: "high",
            EnergyLevel.VERY_HIGH: "high",
        }

        mood_matrix = {
            ("slow", "low"):   MoodCategory.SERENE,
            ("slow", "mid"):   MoodCategory.MELANCHOLY,
            ("slow", "high"):  MoodCategory.MELANCHOLY,
            ("medium", "low"): MoodCategory.CHILL,
            ("medium", "mid"): MoodCategory.GROOVY,
            ("medium", "high"): MoodCategory.GROOVY,
            ("fast", "low"):   MoodCategory.CHILL,
            ("fast", "mid"):   MoodCategory.UPBEAT,
            ("fast", "high"):  MoodCategory.INTENSE,
            ("very_fast", "low"):  MoodCategory.UPBEAT,
            ("very_fast", "mid"):  MoodCategory.INTENSE,
            ("very_fast", "high"): MoodCategory.FRANTIC,
        }

        key = (tempo_speed[tempo], energy_level[energy])
        return mood_matrix.get(key, MoodCategory.GROOVY)

    def describe_meter(self, meter: int) -> MeterDescription:
        """Map beats-per-bar to a meter description.

        Args:
            meter: Number of beats per bar (2, 3, 4, or 6).

        Returns:
            MeterDescription enum value.
        """
        for md in MeterDescription:
            if md.beats == meter:
                return md
        return MeterDescription.COMMON  # Default to 4/4

    def generate_tags(
        self, tempo: TempoMarking, energy: EnergyLevel, mood: MoodCategory
    ) -> List[str]:
        """Generate searchable tags from the semantic analysis.

        These tags can be used for filtering and retrieval in the
        broader Music Maven system.
        """
        tags = [
            tempo.label.lower(),
            energy.label.lower().replace(" ", "-"),
            mood.label.lower(),
        ]
        # Add use-case tags from mood
        tags.extend([uc.strip() for uc in mood.use_cases.split(",")])
        # Add energy descriptors
        tags.extend([d.strip() for d in energy.descriptors.split(",")])
        return tags

    def map(
        self,
        bpm: float,
        meter: int = 4,
        onset_density: Optional[float] = None,
        confidence: float = 1.0,
    ) -> SemanticContext:
        """Perform full semantic mapping from numerical data to LLM context.

        This is the primary method. Takes raw numbers from the beat
        tracking pipeline and produces a complete SemanticContext
        ready for injection into an LLM prompt.

        Args:
            bpm: Detected tempo in beats per minute.
            meter: Detected beats per bar (default 4).
            onset_density: Normalized onset density [0-1] (optional).
            confidence: Confidence score from beat tracker [0-1].

        Returns:
            SemanticContext with all classifications and an LLM prompt string.
        """
        tempo = self.classify_tempo(bpm)
        energy = self.classify_energy(bpm, onset_density)
        mood = self.classify_mood(tempo, energy)
        meter_desc = self.describe_meter(meter)
        tags = self.generate_tags(tempo, energy, mood)

        # Build the LLM prompt string
        llm_prompt = self._build_prompt(
            bpm, tempo, energy, mood, meter_desc, confidence, onset_density
        )

        return SemanticContext(
            bpm=bpm,
            tempo_marking=tempo.label,
            tempo_description=tempo.description,
            energy_level=energy.label,
            energy_descriptors=energy.descriptors,
            mood=mood.label,
            mood_use_cases=mood.use_cases,
            meter=meter,
            meter_description=meter_desc.description,
            confidence=confidence,
            llm_prompt=llm_prompt,
            tags=tags,
        )

    def _build_prompt(
        self,
        bpm: float,
        tempo: TempoMarking,
        energy: EnergyLevel,
        mood: MoodCategory,
        meter_desc: MeterDescription,
        confidence: float,
        onset_density: Optional[float],
    ) -> str:
        """Build a natural language context string for LLM prompt injection.

        This string is designed to be prepended to or embedded within
        an LLM prompt to provide musical context about a track.
        """
        parts = [
            f"[TRACK ANALYSIS]",
            f"Tempo: {bpm:.0f} BPM ({tempo.label} – {tempo.description})",
            f"Energy: {energy.label} ({energy.descriptors})",
            f"Mood: {mood.label}",
            f"Meter: {meter_desc.description}",
            f"Suggested contexts: {mood.use_cases}",
        ]

        if onset_density is not None:
            density_word = (
                "sparse" if onset_density < 0.3
                else "moderate" if onset_density < 0.6
                else "dense"
            )
            parts.append(f"Rhythmic density: {density_word} ({onset_density:.2f})")

        if confidence < 0.5:
            parts.append(
                "Note: Tempo detection confidence is low; "
                "the track may have irregular rhythm or tempo changes."
            )

        return "\n".join(parts)

    def vibe_to_bpm_range(self, vibe_word: str) -> Optional[tuple]:
        """Map a natural language 'vibe' keyword to a BPM range.

        This enables the chatbot to handle queries like:
            "Find me something chill" → (70, 110) BPM range

        Args:
            vibe_word: A mood/energy keyword from the user.

        Returns:
            Tuple of (min_bpm, max_bpm) or None if unrecognized.
        """
        return self.VIBE_TO_BPM.get(vibe_word.lower().strip())

    def find_matching_vibes(self, bpm: float) -> List[str]:
        """Given a BPM, find all vibe words that match.

        Useful for generating descriptions: "At 128 BPM, this track
        could be described as: upbeat, danceable, energetic."
        """
        matches = []
        for vibe, (lo, hi) in self.VIBE_TO_BPM.items():
            if lo <= bpm <= hi:
                matches.append(vibe)
        return matches
