"""
Demo script for PI.4 – Semantic Mapper.
Shows the full pipeline: Audio → Beat Tracking → Semantic Mapping → LLM Context.

Run:
    python demo_pi4.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from app.beat_tracker import BeatTracker
from app.semantic_mapper import SemanticMapper


def plot_semantic_overview(contexts: list, save_path=None, show=True):
    """Create a visual summary of semantic mappings across test files.

    Two-panel figure:
      Left:  BPM vs Energy scatter with mood coloring
      Right: Table of semantic mappings
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [1, 1.3]},
                                    constrained_layout=True)
    fig.suptitle("Semantic Mapping: Tempo → Energy → Mood",
                 fontsize=14, fontweight="bold")

    # ── Left: Scatter plot ──
    mood_colors = {
        "Serene": "#81D4FA",
        "Melancholy": "#B39DDB",
        "Chill": "#80CBC4",
        "Groovy": "#FFD54F",
        "Upbeat": "#FFB74D",
        "Intense": "#E57373",
        "Frantic": "#F44336",
    }

    energy_y = {
        "Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5
    }

    for ctx in contexts:
        color = mood_colors.get(ctx["mood"], "#999999")
        y = energy_y.get(ctx["energy_level"], 3)
        ax1.scatter(ctx["bpm"], y, c=color, s=200, edgecolors="black",
                    linewidth=1.2, zorder=3)
        ax1.annotate(Path(ctx["file"]).stem, (ctx["bpm"], y),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=8)

    ax1.set_xlabel("BPM", fontsize=11)
    ax1.set_ylabel("Energy Level", fontsize=11)
    ax1.set_yticks(list(energy_y.values()))
    ax1.set_yticklabels(list(energy_y.keys()))
    ax1.set_xlim(30, 220)
    ax1.set_ylim(0.5, 5.5)
    ax1.grid(True, alpha=0.3)

    # Legend for mood colors
    patches = [mpatches.Patch(color=c, label=m) for m, c in mood_colors.items()]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, title="Mood")

    # ── Right: Summary table ──
    ax2.axis("off")
    headers = ["File", "BPM", "Tempo", "Energy", "Mood", "Meter"]
    rows = []
    for ctx in contexts:
        rows.append([
            Path(ctx["file"]).stem,
            f"{ctx['bpm']:.1f}",
            ctx["tempo_marking"],
            ctx["energy_level"],
            ctx["mood"],
            f"{ctx['meter']}/4",
        ])

    colors_for_rows = []
    for ctx in contexts:
        c = mood_colors.get(ctx["mood"], "#EEEEEE")
        colors_for_rows.append([c + "40"] * len(headers))  # Light version

    table = ax2.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["#E0E0E0"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_text_props(fontweight="bold")

    ax2.set_title("Semantic Analysis Results", fontsize=12, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vibe_mapping(mapper, save_path=None, show=True):
    """Visualize the vibe-to-BPM mapping as a horizontal bar chart.

    Shows how natural language keywords map to BPM ranges,
    which is key for the chatbot's query understanding.
    """
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    fig.suptitle("Vibe-to-BPM Mapping for Natural Language Queries",
                 fontsize=14, fontweight="bold")

    # Sort by midpoint BPM
    sorted_vibes = sorted(
        mapper.VIBE_TO_BPM.items(),
        key=lambda x: (x[1][0] + x[1][1]) / 2
    )

    labels = [v[0] for v in sorted_vibes]
    starts = [v[1][0] for v in sorted_vibes]
    widths = [v[1][1] - v[1][0] for v in sorted_vibes]

    # Color by energy category
    colors = []
    for s, w in zip(starts, widths):
        mid = s + w / 2
        if mid < 80:
            colors.append("#81D4FA")
        elif mid < 120:
            colors.append("#80CBC4")
        elif mid < 150:
            colors.append("#FFD54F")
        else:
            colors.append("#E57373")

    y_pos = range(len(labels))
    ax.barh(y_pos, widths, left=starts, color=colors, edgecolor="white",
            height=0.7, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("BPM Range", fontsize=11)
    ax.set_xlim(30, 220)
    ax.grid(True, axis="x", alpha=0.3)

    # Add BPM labels on bars
    for i, (s, w) in enumerate(zip(starts, widths)):
        ax.text(s + w / 2, i, f"{s}–{s+w}", ha="center", va="center",
                fontsize=8, fontweight="bold")

    # Add tempo marking regions
    tempo_regions = [
        ("Largo", 40, 60), ("Adagio", 60, 80), ("Andante", 80, 100),
        ("Moderato", 100, 120), ("Allegro", 120, 156), ("Vivace", 156, 176),
        ("Presto", 176, 210),
    ]
    for label, lo, hi in tempo_regions:
        ax.axvspan(lo, hi, alpha=0.06, color="gray")
        ax.text((lo + hi) / 2, len(labels) + 0.3, label,
                ha="center", fontsize=7, style="italic", color="gray")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    print("=" * 65)
    print("  Music Maven G3 – Semantic Mapper Demo (PI.4)")
    print("=" * 65)

    fig_dir = Path("output/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    mapper = SemanticMapper()
    tracker = BeatTracker()
    test_dir = Path("data/test_audio")

    # ── 1. Map beat tracking results to semantic context ──
    print("\n[1] Full pipeline: Audio → Beats → Semantics")

    test_files = {
        "slow_60bpm.wav": None,
        "drums_100bpm.wav": None,
        "sine_120bpm.wav": None,
        "complex_128bpm.wav": None,
        "fast_180bpm.wav": None,
    }

    all_contexts = []

    for fname in test_files:
        path = test_dir / fname
        if not path.exists():
            continue

        # Beat tracking (PI.3)
        bt_result = tracker.track(str(path))

        # Compute onset density from activation function
        if bt_result.beat_activations is not None and len(bt_result.beat_activations) > 0:
            act = bt_result.beat_activations
            onset_density = float(np.mean(act > np.percentile(act, 75)))
        else:
            onset_density = 0.5

        # Semantic mapping (PI.4)
        ctx = mapper.map(
            bpm=bt_result.bpm,
            meter=bt_result.meter,
            onset_density=onset_density,
            confidence=bt_result.confidence,
        )

        ctx_data = ctx.to_dict()
        ctx_data["file"] = fname

        all_contexts.append(ctx_data)

        print(f"\n  {fname}")
        print(f"    BPM: {bt_result.bpm:.1f} → {ctx.tempo_marking} ({ctx.tempo_description})")
        print(f"    Energy: {ctx.energy_level} ({ctx.energy_descriptors})")
        print(f"    Mood: {ctx.mood} → {ctx.mood_use_cases}")
        print(f"    Tags: {ctx.tags[:5]}")

    # ── 2. Print LLM prompt example ──
    print("\n" + "=" * 65)
    print("[2] Example LLM Prompt (for drums_100bpm.wav):")
    print("=" * 65)
    drums_ctx = [c for c in all_contexts if "drums" in c["file"]]
    if drums_ctx:
        print(drums_ctx[0]["llm_prompt"])

    # ── 3. Vibe keyword demo ──
    print("\n[3] Vibe keyword mapping examples:")
    test_vibes = ["chill", "workout", "aggressive", "mellow", "danceable"]
    for vibe in test_vibes:
        bpm_range = mapper.vibe_to_bpm_range(vibe)
        print(f"    '{vibe}' → BPM range {bpm_range}")

    # ── 4. Reverse lookup: BPM → matching vibes ──
    print("\n[4] Reverse vibe lookup:")
    test_bpms = [55, 90, 128, 165, 195]
    for bpm in test_bpms:
        vibes = mapper.find_matching_vibes(bpm)
        print(f"    {bpm} BPM → {vibes}")

    # ── 5. Generate figures ──
    print("\n[5] Generating figures...")
    if all_contexts:
        plot_semantic_overview(
            all_contexts,
            save_path=str(fig_dir / "semantic_overview.png"),
            show=False,
        )

    plot_vibe_mapping(
        mapper,
        save_path=str(fig_dir / "vibe_mapping.png"),
        show=False,
    )

    # ── 6. Save results ──
    json_path = Path("output/semantic_results.json")
    with open(json_path, "w") as f:
        json.dump(all_contexts, f, indent=2)
    print(f"\n[6] Results saved: {json_path}")

    print("\n" + "=" * 65)
    print("  Demo complete. Check output/figures/ for report images.")
    print("=" * 65)


if __name__ == "__main__":
    main()
