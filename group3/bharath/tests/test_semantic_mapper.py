"""
Tests for PI.4 – Semantic Mapper.
Run with: pytest tests/test_semantic_mapper.py -v
"""

import pytest
from app.semantic_mapper import (
    SemanticMapper, SemanticContext,
    TempoMarking, EnergyLevel, MoodCategory, MeterDescription,
)


@pytest.fixture
def mapper():
    return SemanticMapper()


# ── Tempo Classification ──

def test_tempo_largo(mapper):
    assert mapper.classify_tempo(50).label == "Largo"
    print("✓ 50 BPM → Largo")

def test_tempo_adagio(mapper):
    assert mapper.classify_tempo(70).label == "Adagio"
    print("✓ 70 BPM → Adagio")

def test_tempo_andante(mapper):
    assert mapper.classify_tempo(90).label == "Andante"
    print("✓ 90 BPM → Andante")

def test_tempo_moderato(mapper):
    assert mapper.classify_tempo(110).label == "Moderato"
    print("✓ 110 BPM → Moderato")

def test_tempo_allegro(mapper):
    assert mapper.classify_tempo(130).label == "Allegro"
    print("✓ 130 BPM → Allegro")

def test_tempo_vivace(mapper):
    assert mapper.classify_tempo(165).label == "Vivace"
    print("✓ 165 BPM → Vivace")

def test_tempo_presto(mapper):
    assert mapper.classify_tempo(190).label == "Presto"
    print("✓ 190 BPM → Presto")

def test_tempo_prestissimo(mapper):
    assert mapper.classify_tempo(220).label == "Prestissimo"
    print("✓ 220 BPM → Prestissimo")

def test_tempo_edge_low(mapper):
    assert mapper.classify_tempo(20).label == "Largo"
    print("✓ 20 BPM → Largo (edge)")

def test_tempo_edge_high(mapper):
    assert mapper.classify_tempo(300).label == "Prestissimo"
    print("✓ 300 BPM → Prestissimo (edge)")


# ── Energy Classification ──

def test_energy_very_low(mapper):
    result = mapper.classify_energy(45, 0.1)
    assert result == EnergyLevel.VERY_LOW
    print(f"✓ 45 BPM + 0.1 density → {result.label}")

def test_energy_low(mapper):
    result = mapper.classify_energy(70, 0.2)
    assert result == EnergyLevel.LOW
    print(f"✓ 70 BPM + 0.2 density → {result.label}")

def test_energy_moderate(mapper):
    result = mapper.classify_energy(110, 0.4)
    assert result == EnergyLevel.MODERATE
    print(f"✓ 110 BPM + 0.4 density → {result.label}")

def test_energy_high(mapper):
    result = mapper.classify_energy(150, 0.7)
    assert result == EnergyLevel.HIGH
    print(f"✓ 150 BPM + 0.7 density → {result.label}")

def test_energy_very_high(mapper):
    result = mapper.classify_energy(190, 0.9)
    assert result == EnergyLevel.VERY_HIGH
    print(f"✓ 190 BPM + 0.9 density → {result.label}")

def test_energy_without_onset(mapper):
    """Should work with BPM only (no onset density)."""
    result = mapper.classify_energy(130)
    assert result in list(EnergyLevel)
    print(f"✓ 130 BPM (no density) → {result.label}")


# ── Mood Classification ──

def test_mood_serene(mapper):
    tempo = mapper.classify_tempo(50)
    energy = mapper.classify_energy(50, 0.1)
    mood = mapper.classify_mood(tempo, energy)
    assert mood == MoodCategory.SERENE
    print(f"✓ Slow + Low energy → {mood.label}")

def test_mood_upbeat(mapper):
    tempo = mapper.classify_tempo(130)
    energy = mapper.classify_energy(130, 0.5)
    mood = mapper.classify_mood(tempo, energy)
    assert mood == MoodCategory.UPBEAT
    print(f"✓ Fast + Mid energy → {mood.label}")

def test_mood_intense(mapper):
    tempo = mapper.classify_tempo(150)
    energy = mapper.classify_energy(150, 0.8)
    mood = mapper.classify_mood(tempo, energy)
    assert mood == MoodCategory.INTENSE
    print(f"✓ Fast + High energy → {mood.label}")


# ── Meter Description ──

def test_meter_common_time(mapper):
    desc = mapper.describe_meter(4)
    assert "4/4" in desc.description
    print(f"✓ Meter 4 → {desc.description}")

def test_meter_waltz(mapper):
    desc = mapper.describe_meter(3)
    assert "3/4" in desc.description
    print(f"✓ Meter 3 → {desc.description}")

def test_meter_unknown_defaults_to_common(mapper):
    desc = mapper.describe_meter(5)
    assert desc.beats == 4
    print(f"✓ Meter 5 → defaults to common time")


# ── Full Mapping (map method) ──

def test_full_map_returns_context(mapper):
    ctx = mapper.map(bpm=128, meter=4, onset_density=0.65, confidence=0.85)
    assert isinstance(ctx, SemanticContext)
    assert ctx.bpm == 128
    assert ctx.tempo_marking == "Allegro"
    assert len(ctx.llm_prompt) > 0
    assert len(ctx.tags) > 0
    print(f"✓ Full map: {ctx.tempo_marking}, {ctx.energy_level}, {ctx.mood}")

def test_full_map_slow_track(mapper):
    ctx = mapper.map(bpm=55, meter=4, onset_density=0.1)
    assert ctx.tempo_marking == "Largo"
    assert ctx.mood == "Serene"
    print(f"✓ Slow track: {ctx.tempo_marking}, {ctx.mood}")

def test_full_map_fast_track(mapper):
    ctx = mapper.map(bpm=185, meter=4, onset_density=0.85)
    assert ctx.tempo_marking == "Presto"
    assert ctx.energy_level == "Very High"
    print(f"✓ Fast track: {ctx.tempo_marking}, {ctx.energy_level}")

def test_low_confidence_adds_warning(mapper):
    ctx = mapper.map(bpm=100, confidence=0.3)
    assert "low" in ctx.llm_prompt.lower()
    print("✓ Low confidence adds warning to prompt")

def test_context_to_dict(mapper):
    ctx = mapper.map(bpm=120)
    d = ctx.to_dict()
    assert "bpm" in d
    assert "llm_prompt" in d
    assert "tags" in d
    print(f"✓ to_dict() has {len(d)} keys")

def test_context_to_json(mapper):
    ctx = mapper.map(bpm=120)
    j = ctx.to_json()
    import json
    parsed = json.loads(j)
    assert parsed["bpm"] == 120
    print(f"✓ to_json() valid, {len(j)} chars")


# ── Vibe Mapping ──

def test_vibe_chill(mapper):
    result = mapper.vibe_to_bpm_range("chill")
    assert result == (70, 110)
    print(f"✓ 'chill' → {result}")

def test_vibe_workout(mapper):
    result = mapper.vibe_to_bpm_range("workout")
    assert result == (130, 170)
    print(f"✓ 'workout' → {result}")

def test_vibe_unknown(mapper):
    result = mapper.vibe_to_bpm_range("xyzzy")
    assert result is None
    print("✓ Unknown vibe → None")

def test_vibe_case_insensitive(mapper):
    result = mapper.vibe_to_bpm_range("CHILL")
    assert result == (70, 110)
    print("✓ Case insensitive lookup works")

def test_find_matching_vibes(mapper):
    vibes = mapper.find_matching_vibes(128)
    assert "upbeat" in vibes or "energetic" in vibes or "danceable" in vibes
    assert len(vibes) > 0
    print(f"✓ 128 BPM matches vibes: {vibes}")


# ── Tags ──

def test_tags_not_empty(mapper):
    ctx = mapper.map(bpm=128, onset_density=0.6)
    assert len(ctx.tags) > 3
    print(f"✓ Tags generated: {ctx.tags}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
