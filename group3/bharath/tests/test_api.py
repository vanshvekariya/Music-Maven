"""
Tests for PI.1 – FastAPI Beat Tracking Microservice.
Run with: pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from app.main import app, classify_tempo, classify_energy
from app.schemas import TempoClass, EnergyLevel

client = TestClient(app)


# ── Health & Root ──

def test_health_endpoint():
    """Service should respond with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "beat-tracking"
    print(f"✓ Health check: {data}")


def test_root_endpoint():
    """Root should list available endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    print(f"✓ Root info: {data}")


# ── /analyze Endpoint ──

def test_analyze_returns_valid_response():
    """POST /analyze should return a complete BeatTrackingResponse."""
    payload = {
        "file_path": "data/audio/test_song.mp3",
        "include_beats": True,
        "include_downbeats": True,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check all required fields exist
    assert "bpm" in data
    assert "confidence" in data
    assert "tempo_class" in data
    assert "energy_level" in data
    assert "beat_times" in data
    assert "downbeat_times" in data
    assert data["file_path"] == payload["file_path"]
    print(f"✓ Analyze response: BPM={data['bpm']}, class={data['tempo_class']}")


def test_analyze_without_beats():
    """Should respect include_beats=False."""
    payload = {
        "file_path": "data/audio/test_song.mp3",
        "include_beats": False,
        "include_downbeats": False,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["beat_times"] is None
    assert data["downbeat_times"] is None
    print("✓ Beats excluded when not requested")


def test_analyze_missing_file_path():
    """Should return 422 when file_path is missing."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422
    print("✓ Validation correctly rejects missing file_path")


# ── /semantic Endpoint ──

def test_semantic_endpoint():
    """POST /semantic should return LLM-ready descriptors."""
    payload = {"bpm": 140.0, "meter": "4/4", "onset_density": 0.7}
    response = client.post("/semantic", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["tempo_class"] == "Allegro"
    assert "llm_context" in data
    assert "140" in data["llm_context"]
    print(f"✓ Semantic response: {data['llm_context']}")


def test_semantic_low_bpm():
    """Slow tempo should map correctly."""
    payload = {"bpm": 55.0}
    response = client.post("/semantic", json=payload)
    data = response.json()
    assert data["tempo_class"] == "Largo"
    assert data["energy_level"] == "Low Intensity"
    print(f"✓ Low BPM mapping: {data['tempo_class']}, {data['energy_level']}")


def test_semantic_high_bpm():
    """Fast tempo should map to high energy."""
    payload = {"bpm": 185.0, "onset_density": 0.9}
    response = client.post("/semantic", json=payload)
    data = response.json()
    assert data["tempo_class"] == "Presto"
    assert data["energy_level"] == "Very High Intensity"
    print(f"✓ High BPM mapping: {data['tempo_class']}, {data['energy_level']}")


def test_semantic_invalid_bpm():
    """BPM outside range should be rejected."""
    response = client.post("/semantic", json={"bpm": -10})
    assert response.status_code == 422
    response = client.post("/semantic", json={"bpm": 500})
    assert response.status_code == 422
    print("✓ Invalid BPM correctly rejected")


# ── Unit Tests for Helpers ──

def test_classify_tempo_boundaries():
    """Test all tempo classification boundaries."""
    assert classify_tempo(50) == TempoClass.LARGO
    assert classify_tempo(70) == TempoClass.ADAGIO
    assert classify_tempo(90) == TempoClass.ANDANTE
    assert classify_tempo(110) == TempoClass.MODERATO
    assert classify_tempo(130) == TempoClass.ALLEGRO
    assert classify_tempo(165) == TempoClass.VIVACE
    assert classify_tempo(190) == TempoClass.PRESTO
    print("✓ All tempo boundaries correct")


def test_classify_energy_without_onset():
    """Energy classification with BPM only."""
    assert classify_energy(60) == EnergyLevel.LOW
    assert classify_energy(100) == EnergyLevel.MODERATE
    assert classify_energy(140) == EnergyLevel.HIGH
    assert classify_energy(190) == EnergyLevel.VERY_HIGH
    print("✓ Energy classification (BPM-only) correct")


def test_classify_energy_with_onset():
    """Energy classification with onset density."""
    # Low BPM + low density = low energy
    assert classify_energy(60, 0.1) == EnergyLevel.LOW
    # High BPM + high density = very high energy
    assert classify_energy(180, 0.9) == EnergyLevel.VERY_HIGH
    print("✓ Energy classification (BPM + onset) correct")


if __name__ == "__main__":
    """Run tests manually and print results."""
    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_analyze_returns_valid_response,
        test_analyze_without_beats,
        test_analyze_missing_file_path,
        test_semantic_endpoint,
        test_semantic_low_bpm,
        test_semantic_high_bpm,
        test_semantic_invalid_bpm,
        test_classify_tempo_boundaries,
        test_classify_energy_without_onset,
        test_classify_energy_with_onset,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
