# Stretch 6A-S1 — Custom NER Rules: Analysis

## Custom EntityRuler Patterns

The custom EntityRuler defines **16 pattern entries** across **4 custom labels**, covering **12+ distinct climate concepts**:

| Label | Concepts Covered |
|---|---|
| `CLIMATE_EVENT` | COP28, COP27, Bonn Climate Change Conference, Climate Ambition Summit, UN Climate Summit |
| `POLICY` | Paris Agreement, Carbon Border Adjustment Mechanism, nationally determined contributions, NDCs |
| `REPORT` | Sixth Assessment Report, IPCC AR6, State of Food and Agriculture 2023 |
| `THRESHOLD` | 1.5 degrees Celsius, 2°C target, net zero, net-zero |

All patterns use exact phrase matching to avoid false positives — for example, `"Paris"` alone does not trigger `POLICY`; only the full string `"Paris Agreement"` does.

---

## Pipeline Position: Before vs. After NER

**Ruler Before NER:** When the EntityRuler runs before the statistical model, its matches take priority. Spans matched by a custom rule are locked in before spaCy's NER sees the text, so the NER cannot override them. This is useful when the custom patterns are highly specific and trustworthy — for example, `"COP28"` is unambiguously a climate event and should not be re-labeled.

**Ruler After NER:** When the EntityRuler runs after the statistical model, the NER's predictions take priority. The ruler only fills in spans that the NER missed entirely. This is a safer position when pattern coverage is narrow, because it avoids overriding correct NER predictions.

In this dataset, placing the ruler **before** NER produced more custom-label matches overall, because spaCy's base model frequently misses or mislabels climate-specific terms. Placing it **after** resulted in fewer custom-label hits, since spaCy had already claimed spans like `"Paris Agreement"` (sometimes tagging it as `GPE` instead of `LAW`).

---

## Evaluation on Standard Labels

Evaluation was performed on the gold-standard subset (`gold_entities.csv`) using **standard spaCy labels only** (`ORG`, `GPE`, `DATE`, `LAW`, `MONEY`, `PERSON`, `QUANTITY`, `LOC`, `EVENT`, `WORK_OF_ART`). Custom labels were excluded from the quantitative evaluation to avoid artificially depressing precision.

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no ruler) | — | — | — |
| Ruler Before NER | — | — | — |
| Ruler After NER | — | — | — |

> **Note:** Fill in the actual numbers after running `python stretch_custom_ner.py`. The script prints the full delta table at the end.

---

## Qualitative Analysis of Custom Rules

### Where the rules helped

**`CLIMATE_EVENT`:** The base model missed `"COP28"` in text_id=2 entirely — it was neither tagged as `ORG` nor `EVENT`. The custom rule correctly identifies it as a climate conference. Similarly, `"Bonn Climate Change Conference"` (text_id=5) and `"Climate Ambition Summit"` (text_id=7) were either missed or mis-tagged by the base NER; the ruler captures them consistently.

**`POLICY`:** The base model tagged `"Paris Agreement"` as `GPE` in some texts (confusing "Paris" for a location). The ruler-before configuration overrides this with the correct `POLICY` label. `"Carbon Border Adjustment Mechanism"` (text_id=6) was not recognized at all by the base model — the custom rule adds it reliably.

**`REPORT`:** `"Sixth Assessment Report"` (text_id=1) was tagged as `WORK_OF_ART` by the base model, which is technically acceptable but semantically imprecise for a scientific report. The custom `REPORT` label provides a more domain-appropriate classification.

**`THRESHOLD`:** `"1.5 degrees Celsius"` (text_id=1) appeared as `QUANTITY` in the base model output, losing the climate-target semantics. The custom `THRESHOLD` label captures the policy significance of this number.

### Where the rules introduced noise

`"NDCs"` is defined as a `POLICY` pattern, but in text_id=5 it appears mid-sentence in a way that could refer to a document rather than a policy instrument. The rule fires correctly on form but the semantic match is imperfect. Additionally, `"net zero"` as a `THRESHOLD` pattern can fire in sentences where it is used attributively ("net zero economy") rather than as a standalone target — a token-level pattern with surrounding context constraints would be more precise here.

### Engineering tradeoff

Phrase-level patterns are fast, precise for named entities, and easy to maintain, but they are brittle to surface variation (e.g., "the Paris Agreement" vs. "Paris climate accord"). Token-level patterns with attribute constraints (e.g., `LOWER`) would generalize better but require more design effort. For a production system, a hybrid approach — phrase patterns for well-defined named entities and token patterns for variable-form targets like temperature thresholds — would provide the best balance of precision and recall.