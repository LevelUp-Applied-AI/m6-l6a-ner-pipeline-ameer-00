"""
Module 6 Week A — Stretch: Custom NER Rules
stretch_custom_ner.py

Extends the base spaCy NER pipeline with a custom EntityRuler
that captures domain-specific climate terminology.

Run: python stretch_custom_ner.py
"""

import pandas as pd
import spacy
from spacy.pipeline import EntityRuler


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(filepath="data/climate_articles.csv"):
    return pd.read_csv(filepath)


def load_gold(filepath="data/gold_entities.csv"):
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------------
# 2. Define custom EntityRuler patterns
#    - At least 10 distinct entries
#    - At least 3 custom labels
#    - At least 8 distinct concepts
# ---------------------------------------------------------------------------

CUSTOM_PATTERNS = [
    # ── CLIMATE_EVENT (conferences & summits) ───────────────────────────────
    {"label": "CLIMATE_EVENT", "pattern": "COP28"},                          # 1
    {"label": "CLIMATE_EVENT", "pattern": "COP27"},                          # 2
    {"label": "CLIMATE_EVENT", "pattern": "Bonn Climate Change Conference"}, # 3
    {"label": "CLIMATE_EVENT", "pattern": "Climate Ambition Summit"},        # 4
    {"label": "CLIMATE_EVENT", "pattern": "UN Climate Summit"},              # 5

    # ── POLICY (international agreements & legal instruments) ───────────────
    {"label": "POLICY", "pattern": "Paris Agreement"},                       # 6
    {"label": "POLICY", "pattern": "Carbon Border Adjustment Mechanism"},    # 7
    {"label": "POLICY", "pattern": "nationally determined contributions"},   # 8
    {"label": "POLICY", "pattern": "NDCs"},                                  # 9

    # ── REPORT (published scientific or institutional reports) ──────────────
    {"label": "REPORT", "pattern": "Sixth Assessment Report"},               # 10
    {"label": "REPORT", "pattern": "IPCC AR6"},                              # 11
    {"label": "REPORT", "pattern": "State of Food and Agriculture 2023"},    # 12

    # ── THRESHOLD (quantitative climate targets) ────────────────────────────
    {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"},                # 13
    {"label": "THRESHOLD", "pattern": "2°C target"},                         # 14
    {"label": "THRESHOLD", "pattern": "net zero"},                           # 15
    {"label": "THRESHOLD", "pattern": "net-zero"},                           # 16
]

# Summary: 16 entries, 4 labels, covering 12+ distinct concepts ✓


# ---------------------------------------------------------------------------
# 3. Extract entities helper (reusable for any pipeline)
# ---------------------------------------------------------------------------

STANDARD_LABELS = {
    "ORG", "GPE", "DATE", "LAW", "MONEY",
    "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART",
}


def extract_entities(df, nlp):
    """Run nlp pipeline on English texts; return a DataFrame of entities."""
    rows = []
    english_df = df[df["language"] == "en"]
    for _, row in english_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append({
                "text_id":      row["id"],
                "entity_text":  ent.text,
                "entity_label": ent.label_,
                "start_char":   ent.start_char,
                "end_char":     ent.end_char,
            })
    return pd.DataFrame(rows, columns=[
        "text_id", "entity_text", "entity_label", "start_char", "end_char"
    ])


# ---------------------------------------------------------------------------
# 4. Evaluation (standard labels only, to match gold standard)
# ---------------------------------------------------------------------------

def evaluate_ner(predicted_df, gold_df):
    """Precision / Recall / F1 on standard labels only."""
    # Keep only standard-label predictions so custom labels don't depress precision
    pred_standard = predicted_df[predicted_df["entity_label"].isin(STANDARD_LABELS)]

    pred_set = set(zip(
        pred_standard["text_id"],
        pred_standard["entity_text"],
        pred_standard["entity_label"],
    ))
    gold_set = set(zip(
        gold_df["text_id"],
        gold_df["entity_text"],
        gold_df["entity_label"],
    ))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# 5. Build pipelines
# ---------------------------------------------------------------------------

def build_pipeline_ruler_before():
    """EntityRuler inserted BEFORE the statistical NER (ruler takes priority)."""
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(CUSTOM_PATTERNS)
    return nlp


def build_pipeline_ruler_after():
    """EntityRuler inserted AFTER the statistical NER (NER takes priority)."""
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(CUSTOM_PATTERNS)
    return nlp


# ---------------------------------------------------------------------------
# 6. Qualitative examples for custom labels
# ---------------------------------------------------------------------------

def show_custom_label_examples(entities_df, label, n=3):
    """Print example matches for a given custom label."""
    hits = entities_df[entities_df["entity_label"] == label]
    print(f"\n  [{label}] — {len(hits)} matches found")
    for _, row in hits.head(n).iterrows():
        print(f"    text_id={row['text_id']} | '{row['entity_text']}'")


# ---------------------------------------------------------------------------
# 7. Main comparison
# ---------------------------------------------------------------------------

def main():
    df   = load_data()
    gold = load_gold()

    english_df = df[df["language"] == "en"]

    # ── Baseline (no EntityRuler) ──────────────────────────────────────────
    print("=" * 60)
    print("BASELINE — spaCy en_core_web_sm (no EntityRuler)")
    print("=" * 60)
    nlp_base = spacy.load("en_core_web_sm")
    base_entities = extract_entities(df, nlp_base)

    print(f"Total entities extracted : {len(base_entities)}")
    print(f"Label breakdown          :")
    for label, count in base_entities["entity_label"].value_counts().items():
        print(f"  {label:<20} {count}")

    base_metrics = evaluate_ner(base_entities, gold)
    print(f"\nEvaluation vs gold (standard labels only):")
    print(f"  Precision : {base_metrics['precision']}")
    print(f"  Recall    : {base_metrics['recall']}")
    print(f"  F1        : {base_metrics['f1']}")
    print(f"  TP={base_metrics['tp']}  FP={base_metrics['fp']}  FN={base_metrics['fn']}")

    # ── EntityRuler BEFORE NER ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RULER BEFORE NER — custom rules take priority")
    print("=" * 60)
    nlp_before = build_pipeline_ruler_before()
    before_entities = extract_entities(df, nlp_before)

    standard_before = before_entities[before_entities["entity_label"].isin(STANDARD_LABELS)]
    custom_before   = before_entities[~before_entities["entity_label"].isin(STANDARD_LABELS)]

    print(f"Total entities extracted : {len(before_entities)}")
    print(f"  Standard-label entities: {len(standard_before)}")
    print(f"  Custom-label entities  : {len(custom_before)}")
    print(f"Label breakdown          :")
    for label, count in before_entities["entity_label"].value_counts().items():
        print(f"  {label:<30} {count}")

    before_metrics = evaluate_ner(before_entities, gold)
    print(f"\nEvaluation vs gold (standard labels only):")
    print(f"  Precision : {before_metrics['precision']}")
    print(f"  Recall    : {before_metrics['recall']}")
    print(f"  F1        : {before_metrics['f1']}")
    print(f"  TP={before_metrics['tp']}  FP={before_metrics['fp']}  FN={before_metrics['fn']}")

    print("\nCustom label examples (ruler-before):")
    for lbl in ["CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD"]:
        show_custom_label_examples(before_entities, lbl)

    # ── EntityRuler AFTER NER ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RULER AFTER NER — statistical NER takes priority")
    print("=" * 60)
    nlp_after = build_pipeline_ruler_after()
    after_entities = extract_entities(df, nlp_after)

    standard_after = after_entities[after_entities["entity_label"].isin(STANDARD_LABELS)]
    custom_after   = after_entities[~after_entities["entity_label"].isin(STANDARD_LABELS)]

    print(f"Total entities extracted : {len(after_entities)}")
    print(f"  Standard-label entities: {len(standard_after)}")
    print(f"  Custom-label entities  : {len(custom_after)}")
    print(f"Label breakdown          :")
    for label, count in after_entities["entity_label"].value_counts().items():
        print(f"  {label:<30} {count}")

    after_metrics = evaluate_ner(after_entities, gold)
    print(f"\nEvaluation vs gold (standard labels only):")
    print(f"  Precision : {after_metrics['precision']}")
    print(f"  Recall    : {after_metrics['recall']}")
    print(f"  F1        : {after_metrics['f1']}")
    print(f"  TP={after_metrics['tp']}  FP={after_metrics['fp']}  FN={after_metrics['fn']}")

    print("\nCustom label examples (ruler-after):")
    for lbl in ["CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD"]:
        show_custom_label_examples(after_entities, lbl)

    # ── Delta summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DELTA SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<12} {'Baseline':>10} {'Before NER':>12} {'After NER':>11}")
    print("-" * 48)
    for metric in ["precision", "recall", "f1"]:
        b  = base_metrics[metric]
        bf = before_metrics[metric]
        af = after_metrics[metric]
        print(f"{metric:<12} {b:>10.4f} {bf:>12.4f} {af:>11.4f}")

    print(f"\n{'':12} {'Baseline':>10} {'Before NER':>12} {'After NER':>11}")
    print("-" * 48)
    print(f"{'Total ents':<12} {len(base_entities):>10} "
          f"{len(before_entities):>12} {len(after_entities):>11}")
    print(f"{'Custom ents':<12} {'0':>10} "
          f"{len(custom_before):>12} {len(custom_after):>11}")


if __name__ == "__main__":
    main()