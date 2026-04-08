"""
Demo script for PI.1 – shows the API working end-to-end.
Run the server first:  uvicorn app.main:app --reload --port 8000
Then run this:         python demo_pi1.py
"""

import httpx
import json

BASE = "http://localhost:8000"


def demo():
    print("=" * 60)
    print("  Music Maven G3 – Beat Tracking API Demo (PI.1)")
    print("=" * 60)

    # 1. Health check
    print("\n[1] GET /health")
    r = httpx.get(f"{BASE}/health")
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(r.json(), indent=4)}")

    # 2. Analyze a song
    print("\n[2] POST /analyze")
    payload = {
        "file_path": "data/audio/music4all/00123.mp3",
        "include_beats": True,
        "include_downbeats": True,
    }
    print(f"    Request: {json.dumps(payload, indent=4)}")
    r = httpx.post(f"{BASE}/analyze", json=payload)
    data = r.json()
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(data, indent=4)}")

    # 3. Semantic mapping
    print("\n[3] POST /semantic")
    payload = {"bpm": 128.0, "meter": "4/4", "onset_density": 0.65}
    print(f"    Request: {json.dumps(payload, indent=4)}")
    r = httpx.post(f"{BASE}/semantic", json=payload)
    data = r.json()
    print(f"    Status: {r.status_code}")
    print(f"    Response: {json.dumps(data, indent=4)}")
    print(f"\n    LLM Context String:")
    print(f"    → \"{data['llm_context']}\"")

    # 4. Edge case: very fast song
    print("\n[4] POST /semantic (Presto)")
    payload = {"bpm": 200.0, "onset_density": 0.95}
    r = httpx.post(f"{BASE}/semantic", json=payload)
    data = r.json()
    print(f"    → {data['tempo_class']} | {data['energy_level']}")

    # 5. Edge case: very slow song
    print("\n[5] POST /semantic (Largo)")
    payload = {"bpm": 50.0, "onset_density": 0.1}
    r = httpx.post(f"{BASE}/semantic", json=payload)
    data = r.json()
    print(f"    → {data['tempo_class']} | {data['energy_level']}")

    # 6. Swagger docs link
    print(f"\n[6] Interactive API docs available at: {BASE}/docs")

    print("\n" + "=" * 60)
    print("  Demo complete. All endpoints functional.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
