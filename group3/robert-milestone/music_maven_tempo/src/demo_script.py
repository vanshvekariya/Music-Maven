import pandas as pd
from src.query_engine import TemporalQueryEngine

df = pd.read_csv("outputs/tempo_results_full.csv")
engine = TemporalQueryEngine(df)

print("\n=== WORKOUT TRACKS ===")
print(engine.workout_tracks()[["song", "tempo_consensus", "confidence"]])

print("\n=== SIMILAR GROOVE BUT SLOWER ===")
print(engine.similar_groove_but_slower("blues.00000"))