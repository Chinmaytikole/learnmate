# LearnMate Failure-Case & Rescue Gallery

Total evaluated records: **2**

## 1. Error & Outcome Taxonomy

| Category | Count | Share (%) | Description |
|---|---:|---:|---|
| `nli_rescue` | 0 | 0.0% | NLI Rescue (Baseline answered unsupportably/wrongly; Proposed correctly abstained) |
| `false_answer_unanswerable` | 0 | 0.0% | False Answer on Unanswerable Question (Hallucination / Failed Abstention) |
| `unsupported_citation` | 0 | 0.0% | Unsupported Answer / Citation Miss (Answered, but failed to cite gold passage) |
| `false_abstention` | 2 | 100.0% | False Abstention (Over-conservative refusal on an answerable question) |
| `wrong_span_supported` | 0 | 0.0% | Extraction / Span Error (Gold passage cited, but answer token F1 < threshold) |
| `correct` | 0 | 0.0% | Grounded Correct (Accurate & supported answer, or proper abstention) |

## 2. Representative Examples

### NLI Rescue (Baseline answered unsupportably/wrongly; Proposed correctly abstained) (`nli_rescue`)

_No examples observed in this category._

### False Answer on Unanswerable Question (Hallucination / Failed Abstention) (`false_answer_unanswerable`)

_No examples observed in this category._

### Unsupported Answer / Citation Miss (Answered, but failed to cite gold passage) (`unsupported_citation`)

_No examples observed in this category._

### False Abstention (Over-conservative refusal on an answerable question) (`false_abstention`)

**Case 1** (`qid=56cdaf0362d2951400fa6808`)
- **Question:** Which company made Spectre?
- **Answerable:** `True` | **Gold answers:** `['Eon Productions']`
- **Baseline prediction:** `NO_ANSWER`
- **System prediction:** `NO_ANSWER` (abstained=`True`)
- **Signals:** top_sim=`0.759`, agreement=`1.0`, confidence=`0.0`
- **Citations:** predicted=`['63b0ca14faff30af']` vs gold=`63b0ca14faff30af`
- **Diagnosis:** Over-conservative abstention on answerable question despite retrieved context.

**Case 2** (`qid=56cdaf0362d2951400fa6809`)
- **Question:** Who is the star of Spectre?
- **Answerable:** `True` | **Gold answers:** `['Daniel Craig']`
- **Baseline prediction:** `NO_ANSWER`
- **System prediction:** `NO_ANSWER` (abstained=`True`)
- **Signals:** top_sim=`0.749`, agreement=`1.0`, confidence=`0.0`
- **Citations:** predicted=`['63b0ca14faff30af']` vs gold=`63b0ca14faff30af`
- **Diagnosis:** Over-conservative abstention on answerable question despite retrieved context.

### Extraction / Span Error (Gold passage cited, but answer token F1 < threshold) (`wrong_span_supported`)

_No examples observed in this category._
