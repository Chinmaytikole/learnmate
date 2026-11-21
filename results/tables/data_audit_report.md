# LearnMate: SQuAD 2.0 Data Audit Report

**Group:** G06  
**Status:** Clean, Zero-Leakage Article-Level Partition  

## 1. Split Counts and Distributions

| Split   |   Articles |   Unique Passages |   Total Questions |   Answerable | Answerable (%)   |   Unanswerable | Unanswerable (%)   |   Ans Len Mean (Words) |   Ans Len Median (Words) |   Passage Len Mean (Words) |   Passage Len Median (Words) |
|:--------|-----------:|------------------:|------------------:|-------------:|:-----------------|---------------:|:-------------------|-----------------------:|-------------------------:|---------------------------:|-----------------------------:|
| Train   |        354 |             15047 |            102698 |        68147 | 66.4%            |          34551 | 33.6%              |                    3.3 |                        2 |                      119.1 |                          110 |
| Calib   |         44 |              2072 |             14372 |         9787 | 68.1%            |           4585 | 31.9%              |                    3   |                        2 |                      117.6 |                          108 |
| Tune    |         44 |              1910 |             13249 |         8887 | 67.1%            |           4362 | 32.9%              |                    3.4 |                        2 |                      124.6 |                          114 |
| Dev     |         35 |              1204 |             11873 |         5928 | 49.9%            |           5945 | 50.1%              |                    3.2 |                        2 |                      129.3 |                          115 |

## 2. Question-Type Distribution (By First Word)

| Question Type | Train (%) | Calib (%) | Tune (%) | Dev (%) |
|---|---|---|---|---|
| Can/could | 0.1% | 0.0% | 0.0% | 0.1% |
| Did/do/does | 0.4% | 0.3% | 0.2% | 0.2% |
| How | 9.1% | 7.7% | 9.3% | 9.5% |
| In | 5.1% | 5.5% | 4.1% | 4.4% |
| Is/are/was/were | 0.5% | 0.5% | 0.2% | 0.3% |
| Name | 0.1% | 0.1% | 0.1% | 0.1% |
| Other | 13.7% | 18.3% | 12.5% | 13.8% |
| What | 46.0% | 43.1% | 46.8% | 49.1% |
| When | 6.1% | 5.6% | 6.5% | 6.2% |
| Where | 3.6% | 3.0% | 4.6% | 3.6% |
| Which | 4.4% | 5.4% | 3.1% | 2.8% |
| Who | 9.6% | 9.4% | 10.5% | 8.4% |
| Why | 1.4% | 1.0% | 2.0% | 1.6% |

## 3. Leakage Verification Results

- **Overall Status:** PASSED (Zero Leakage)
- ✓ **Article overlap (train vs calib)**: overlap = 0
- ✓ **Article overlap (train vs tune)**: overlap = 0
- ✓ **Article overlap (calib vs tune)**: overlap = 0
- ✓ **Article overlap (train vs dev)**: overlap = 0
- ✓ **Article overlap (calib vs dev)**: overlap = 0
- ✓ **Article overlap (tune vs dev)**: overlap = 0
- ✓ **QID overlap (train vs calib)**: overlap = 0
- ✓ **QID overlap (train vs tune)**: overlap = 0
- ✓ **QID overlap (train vs dev)**: overlap = 0
- ✓ **QID overlap (calib vs tune)**: overlap = 0
- ✓ **QID overlap (calib vs dev)**: overlap = 0
- ✓ **QID overlap (tune vs dev)**: overlap = 0

## 4. Documented Dataset Limitations (For Research Paper)

1. Domain Bias: SQuAD 2.0 covers Wikipedia articles only, which are encyclopedic, dense, and third-person, differing from conversational or pedagogical course materials.
2. Adversarial Unanswerable Design: Negatives were crowdsourced by Workers looking at passages to write similar-sounding unanswerable questions; models can exploit lexical artifact cues.
3. Reference Redundancy: Training examples contain exactly 1 reference answer, whereas dev examples contain ~3 plausible reference spans.
4. Retrieval Gold Standard: No separate human-annotated retrieval ground truth exists; the context paragraph hosting the QA pair is used as the silver gold-standard passage.
5. Course Evaluation Generalizability: The course evaluation set is instructor-specific, smaller in scale, and LLM-synthesized before human blind annotation.