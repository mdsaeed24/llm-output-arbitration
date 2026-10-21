# Evaluation: multi-critic arbitration vs a single reviewer

> Generated 2026-10-21 08:12 UTC from run `reports/runs/20261021-080058` · 40 cases (10 clean, 30 with 35 hand-planted issues) · total spend $0.287

## Headline

**The multi-critic system caught 94% (33/35) of planted errors vs 97% (34/35) for a single reviewer, with 23 false flags vs 6 (0.57 vs 0.15 per case).**

- **Recall:** the full system did **not** beat the single reviewer (paired exact McNemar p = 1.000; system-only catches 0, baseline-only catches 1). With n = 35 planted issues this difference is **not** statistically significant at α = 0.05.
- **False flags:** the system raised more false flags (23 vs 6; ≥ medium severity: 21 vs 2). Precision 59% vs 86%.
- **Sensitivity:** 11 of the system's false flags are omission-style complaints anchored on a planted error's span (mostly the completeness critic restating a planted error). Counting those as duplicates instead: 12 vs 6 false flags, precision 73% vs 86% — the false-flag comparison does not change direction.
- **Verdicts:** pass/fail accuracy 95% vs 98% (paired p = 1.000). Clean outputs passed: 10/10 vs 10/10; flawed outputs failed: 28/30 vs 29/30.
- **Cost & latency:** $0.0060 vs $0.0012 per case (5.0× the cost) and 31.2s vs 4.8s critical-path LLM time per case.

## Results

Every row except the single reviewer is computed from the SAME system run (ablations re-use the critics' stored outputs), so they see identical critic findings.

| Variant | Recall [95% CI] | Factual | Logic | Completeness | Precision | False flags (≥ med) | Clean cases flagged | Verdict accuracy | $ / case | s / case |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full system** (3 critics + adjudicator) | 94% (33/35) [81%–98%] | 87% (13/15) | 100% (10/10) | 100% (10/10) | 59% (33/56) | 23 (21) | 0/10 | 95% (38/40) [83%–99%] | 0.0060 | 31.2 |
| **Single reviewer** (deepseek-v4-pro) | 97% (34/35) [85%–99%] | 93% (14/15) | 100% (10/10) | 100% (10/10) | 86% (37/43) | 6 (2) | 0/10 | 98% (39/40) [87%–100%] | 0.0012 | 4.8 |
| Ablation: union of critics, no adjudicator | 94% (33/35) [81%–98%] | 87% (13/15) | 100% (10/10) | 100% (10/10) | 52% (33/64) | 31 (24) | 1/10 | 95% (38/40) [83%–99%] | 0.0031 | 24.8 |
| Ablation: factual critic alone (deepseek-v4-pro) | 60% (21/35) [44%–74%] | 87% (13/15) | 80% (8/10) | 0% (0/10) | 88% (21/24) | 3 (1) | 0/10 | 70% (28/40) [55%–82%] | 0.0016 | 5.6 |
| Ablation: logic critic alone (gemini-3.5-flash-lite) | 49% (17/35) [33%–64%] | 47% (7/15) | 100% (10/10) | 0% (0/10) | 100% (17/17) | 0 (0) | 0/10 | 68% (27/40) [52%–80%] | 0.0010 | 24.3 |
| Ablation: completeness critic alone (deepseek-flash) | 29% (10/35) [16%–45%] | 0% (0/15) | 0% (0/10) | 100% (10/10) | 26% (10/38) | 28 (23) | 1/10 | 82% (33/40) [68%–91%] | 0.0005 | 2.7 |

*Recall per type uses the PLANTED type. False flags = findings that match no planted issue (every finding on a clean case is one). s / case = critical-path LLM time (slowest critic + adjudicator), excluding queueing and retry waits.*

## Sensitivity: completeness restatements of planted errors

The strict rule (agreed before the run) never lets a completeness finding match a factual/logic planted issue. A completeness critic that restates a planted error as an omission ("the summary reverses the core distinction…") is therefore a false flag under the strict rule, although it is really a duplicate. This table shows the same results with those restatements — omission findings whose quote overlaps a planted factual/logic span — set aside. It is a mechanical proxy (an anchored complaint could occasionally be about something else — check the false-flag lists) and is NOT the headline.

| Variant | False flags (strict) | Restatements of planted errors | False flags excl. restatements | Precision (strict) | Precision excl. restatements |
|---|---|---|---|---|---|
| system | 23 | 11 | 12 | 59% | 73% |
| baseline | 6 | 0 | 6 | 86% | 86% |
| union | 31 | 15 | 16 | 52% | 67% |
| critic:factual | 3 | 0 | 3 | 88% | 88% |
| critic:logic | 0 | 0 | 0 | 100% | 100% |
| critic:completeness | 28 | 15 | 13 | 26% | 43% |

## Paired comparisons

**Full system (3 critics + adjudicator) vs Single reviewer (deepseek-v4-pro)**

| | count |
|---|---|
| planted issues caught by both | 33 |
| caught only by system | 0 |
| caught only by baseline | 1 |
| caught by neither | 1 |
| exact McNemar p (recall) | 1.000 |
| cases where only system had the right verdict | 0 |
| cases where only baseline had the right verdict | 1 |
| exact McNemar p (verdicts) | 1.000 |

**Full system (3 critics + adjudicator) vs Ablation: union of critics, no adjudicator**

| | count |
|---|---|
| planted issues caught by both | 33 |
| caught only by system | 0 |
| caught only by union | 0 |
| caught by neither | 2 |
| exact McNemar p (recall) | 1.000 |
| cases where only system had the right verdict | 0 |
| cases where only union had the right verdict | 0 |
| exact McNemar p (verdicts) | 1.000 |

## How matches were made

Completeness matches come from the LLM judge (deepseek-flash); factual/logic matches from span overlap (≥ 50% of the shorter span). The judge saw descriptions only, never which variant raised them.

| Variant | Planted caught via span | via judge |
|---|---|---|
| system | 23 | 10 |
| baseline | 24 | 10 |
| union | 23 | 10 |
| critic:factual | 21 | 0 |
| critic:logic | 17 | 0 |
| critic:completeness | 0 | 10 |

Judge decisions: 20 complaint(s) matched, 28 rejected · judge cost $0.0000 (not charged to any variant).

### Judge matches — spot-check these

| Case | Planted omission | Complaint | Judge's reasoning | Raised by |
|---|---|---|---|---|
| `amoxicillin-alcohol-missed` | A direct answer about alcohol: moderate drinking has no known direct interaction with amo… | The output never answers whether alcohol can be consumed with amoxicillin, which is the entire question asked. It shoul… | This complaint directly identifies the missing answer about alcohol with amoxicillin: moderate drinking is generally no… | system, union, completeness |
| `amoxicillin-alcohol-missed` | A direct answer about alcohol: moderate drinking has no known direct interaction with amo… | The output does not answer the user's question about drinking alcohol while taking amoxicillin. It provides general dru… | This complaint also points to the failure to answer whether alcohol can be consumed with amoxicillin, which is the same… | baseline |
| `api-word-limit-ignored` | Compliance with the explicit 'under 50 words' limit; the answer runs to well over 100 wor… | The output ignores the explicit 'under 50 words' constraint; the whole answer is roughly 110 words, more than double th… | The complaint explicitly identifies the failure to comply with the 'under 50 words' limit, noting the answer is roughly… | system, union, completeness |
| `api-word-limit-ignored` | Compliance with the explicit 'under 50 words' limit; the answer runs to well over 100 wor… | The output exceeds the 50-word limit specified in the original question. The entire response is approximately 100 words… | This complaint also directly points to the violation of the 50-word limit, stating the response is approximately 100 wo… | baseline |
| `process-thread-mixed` | An advantage of processes (e.g. isolation: a crash or memory corruption in one process do… | The output gives an advantage of threads but never gives an advantage of a process, which the question explicitly reque… | The complaint states the answer fails to give an advantage of a process, which is exactly the missing item in the answe… | system, union, completeness |
| `process-thread-mixed` | An advantage of processes (e.g. isolation: a crash or memory corruption in one process do… | The original question asks for one advantage of each (process and thread). The output gives an advantage of threads but… | This complaint also identifies the missing advantage of processes, matching the same answer-key omission. | baseline |
| `recursion-example-ignored` | A code example of a recursive function, which the question explicitly asked for. | The output never provides a code example, which the question explicitly requested ("give a code example"). A short func… | The complaint explicitly states the output lacks a code example, which the question requested, matching the omission of… | system, union, completeness |
| `recursion-example-ignored` | A code example of a recursive function, which the question explicitly asked for. | The original question asks for a code example, but the output provides none. A code example (e.g., factorial or Fibonac… | This complaint also identifies the missing code example (e.g., factorial or Fibonacci), directly corresponding to the a… | baseline |
| `reverse-string-complexity-skipped` | The time complexity of the function (O(n) in the length of the string), which the questio… | The output never states the time complexity of the reversal, which the question explicitly requested. It should say the… | The complaint says the output never states the time complexity and should say O(n) linear in string length, which is ex… | system, union, completeness |
| `reverse-string-complexity-skipped` | The time complexity of the function (O(n) in the length of the string), which the questio… | The original question asks to 'state its time complexity', but the output never mentions time complexity at all. This i… | This complaint also points to the missing statement of time complexity, matching the same omission. | baseline |
| `student-loan-no-recommendation` | An actual recommendation or decision rule for the user's case (e.g. build an emergency fu… | The user asked 'Should I...?' — a direct request for a recommendation — but the output never commits to an answer or gi… | This complaint directly states the output never commits to an answer or gives a decision rule and should state which op… | system, union, completeness |
| `student-loan-no-recommendation` | An actual recommendation or decision rule for the user's case (e.g. build an emergency fu… | The user asked 'Should I pay off my 7% student loan or invest it in an index fund?' The output does not provide a recom… | This complaint says the output provides no recommendation or clear decision framework tailored to the user and instead … | baseline |
| `tcp-udp-analogy-ignored` | A real-world analogy for the TCP/UDP difference (e.g. registered mail vs postcards), whic… | The output never provides a real-world analogy, which is the explicit requirement of the question. It only gives a tech… | The complaint explicitly states the output lacks a real-world analogy, which is exactly the missing item in the answer … | system, union, completeness |
| `tcp-udp-analogy-ignored` | A real-world analogy for the TCP/UDP difference (e.g. registered mail vs postcards), whic… | The original question asks to explain the difference using a real-world analogy, but the output contains no analogy at … | This complaint also identifies the absence of any real-world analogy, matching the same answer-key omission. | baseline |
| `us-government-veto-skipped` | How Congress overrides a presidential veto: a two-thirds vote in both the House and the S… | The output never answers the second half of the question: how Congress can override a presidential veto. It should expl… | The complaint explicitly states the text fails to explain the veto override process and specifies a two-thirds vote in … | system, union, completeness |
| `us-government-veto-skipped` | How Congress overrides a presidential veto: a two-thirds vote in both the House and the S… | The output does not address the second part of the question: how Congress can override a presidential veto. The answer … | This complaint also identifies the missing veto override explanation and names the two-thirds majority in both chambers… | baseline |
| `water-ethanol-why-skipped` | An explanation of why the boiling points differ: water forms a more extensive hydrogen-bo… | The output never explains why the boiling points differ. The question explicitly asks for an explanation; a competent a… | The complaint states the output never explains why the boiling points differ and specifically invokes water's stronger … | system, union, completeness |
| `water-ethanol-why-skipped` | An explanation of why the boiling points differ: water forms a more extensive hydrogen-bo… | The question asks to explain why the boiling points differ, but the output only describes practical consequences and ne… | This complaint also says the output fails to give the reason for the difference, citing hydrogen bonding in water versu… | baseline |
| `ww2-end-mixed` | Any discussion of the role the atomic bombs on Hiroshima and Nagasaki (6 and 9 August 194… | The output never addresses the third part of the question: the role the atomic bombs played in Japan's surrender. It sh… | The complaint explicitly states the output never addresses the role of the atomic bombs in Japan's surrender, which is … | system, union, completeness |
| `ww2-end-mixed` | Any discussion of the role the atomic bombs on Hiroshima and Nagasaki (6 and 9 August 194… | The question asks about the role of atomic bombs in Japan's surrender, but the output does not mention the atomic bombs… | This complaint also identifies the absence of any discussion of the atomic bombs' role in Japan's surrender, matching m… | baseline |

## Critic analytics

| Critic | Model | Valid issues | Hallucinated quotes | True positives (precision) | Planted caught alone | Unique catches | Overruled by adjudicator | …of which really false |
|---|---|---|---|---|---|---|---|---|
| factual | deepseek-v4-pro | 24 | 0 (0%) | 21 (88%) | 21 | 6 | 2/24 (8%) | 2/2 |
| logic | gemini-3.5-flash-lite | 17 | 0 (0%) | 17 (100%) | 17 | 2 | 0/17 (0%) | 0/0 |
| completeness | deepseek-flash | 38 | 0 (0%) | 10 (26%) | 10 | 10 | 6/38 (16%) | 6/6 |

- **Most valid (true-positive) issues:** factual (21).
- **Overruled most often:** completeness (6/38 = 16% of its adjudicated issues).

**Agreement between critics** (pass/fail verdict each critic would give alone):

| Pair | Agree |
|---|---|
| factual – logic | 88% (35/40) |
| factual – completeness | 68% (27/40) |
| logic – completeness | 65% (26/40) |
| all critics | 60% (24/40) |

Issue groups raised by 2+ critics: 23% (15/64). Low corroboration is expected by design: each critic owns one dimension, so critics rarely flag the same span — single-source issues are what the adjudicator exists to check.

## What the adjudicator changed

- Cases adjudicated: 31; short-circuited (unanimous confident pass): 9, of which really clean: 9; fell back to union after an adjudicator error: 0.
- Issue groups ruled on: 64 (56 confirmed, 8 dismissed).
- Dismissals that removed a false flag: 8/8; dismissals that threw away a real catch: 0/8.
- Pass/fail verdicts the adjudicator fixed vs the raw union: 0; broke: 0.

## Misses

### Missed by both (hardest cases)

- `mortgage-total-interest` — factual: "total interest of about $131,676" — Total interest is about $231,676 ($431,676 paid minus the $200,000 principal), not $131,6…

### Missed by the full system but caught by the baseline

- `everest-boiling-point` — factual: "boils at around 50°C" — Water boils at roughly 70°C on the summit of Everest, not 50°C.

### Missed by the baseline but caught by the full system

- none

## False flags on clean cases

None — neither raised any issue on a clean case.

## Per-case results

| Case | Tags | Planted | System caught | Baseline caught | System FF | Baseline FF | System verdict | Baseline verdict |
|---|---|---|---|---|---|---|---|---|
| `clean-photosynthesis` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-french-revolution` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-python-mutable-default` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-compound-interest` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-vitamin-d` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-sql-left-join` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-apollo-11` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-boiled-eggs` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-binary-search` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `clean-rates-inflation` | clean | 0 | 0/0 | 0/0 | 0 | 0 | pass ✓ | pass ✓ |
| `berlin-wall-reunification-date` | wrong-date | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `penicillin-discovery-date` | wrong-date | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `python3-release-date` | wrong-date, subtle | 1 | 1/1 | 1/1 | 0 | 1 | fail ✓ | fail ✓ |
| `everest-boiling-point` | wrong-number | 1 | 0/1 | 1/1 | 0 | 0 | pass ✗ | fail ✓ |
| `adult-bone-count` | wrong-number | 1 | 1/1 | 1/1 | 1 | 0 | fail ✓ | fail ✓ |
| `mortgage-total-interest` | wrong-number, subtle | 1 | 0/1 | 0/1 | 0 | 0 | pass ✗ | pass ✗ |
| `south-pole-scott-name` | wrong-name | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `dna-photo-51-name` | wrong-name | 1 | 1/1 | 1/1 | 2 | 0 | fail ✓ | fail ✓ |
| `github-acquirer-name` | wrong-name, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `coffee-caffeine-amount` | wrong-number, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `tomato-fruit-contradiction` | contradiction | 1 | 1/1 | 1/1 | 1 | 0 | fail ✓ | fail ✓ |
| `roth-ira-contradiction` | contradiction | 1 | 1/1 | 1/1 | 1 | 1 | fail ✓ | fail ✓ |
| `list-dict-key-contradiction` | contradiction, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `redesign-sales-unsupported` | unsupported-conclusion | 1 | 1/1 | 1/1 | 3 | 0 | fail ✓ | fail ✓ |
| `coffee-longevity-unsupported` | unsupported-conclusion | 1 | 1/1 | 1/1 | 2 | 1 | fail ✓ | fail ✓ |
| `active-fund-unsupported` | unsupported-conclusion, subtle | 1 | 1/1 | 1/1 | 2 | 1 | fail ✓ | fail ✓ |
| `english-science-circular` | circular | 1 | 1/1 | 1/1 | 2 | 0 | fail ✓ | fail ✓ |
| `unit-tests-circular` | circular, subtle | 1 | 1/1 | 1/1 | 1 | 0 | fail ✓ | fail ✓ |
| `us-government-veto-skipped` | skipped-part | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `water-ethanol-why-skipped` | skipped-part | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `reverse-string-complexity-skipped` | skipped-part | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `student-loan-no-recommendation` | missed-point | 1 | 1/1 | 1/1 | 2 | 1 | fail ✓ | fail ✓ |
| `amoxicillin-alcohol-missed` | missed-point | 1 | 1/1 | 1/1 | 1 | 0 | fail ✓ | fail ✓ |
| `api-word-limit-ignored` | ignored-constraint | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `recursion-example-ignored` | ignored-constraint | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `tcp-udp-analogy-ignored` | ignored-constraint | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `ww2-end-mixed` | mixed, wrong-name, skipped-part | 2 | 2/2 | 2/2 | 0 | 0 | fail ✓ | fail ✓ |
| `newton-laws-mixed` | mixed, wrong-date, contradiction | 2 | 2/2 | 2/2 | 1 | 0 | fail ✓ | fail ✓ |
| `process-thread-mixed` | mixed, wrong-claim, skipped-part | 2 | 2/2 | 2/2 | 0 | 0 | fail ✓ | fail ✓ |
| `index-fund-mixed` | mixed, wrong-name, wrong-number, unsupported-conclusion | 3 | 3/3 | 3/3 | 4 | 1 | fail ✓ | fail ✓ |

## Run health

- Cases with a crashed run: 0
- Degraded verdicts (a critic or the adjudicator failed): 0
- Critic failures after retries: 0
- Failed HTTP attempts that were retried: 2

## Caveats

- **Small sample.** 35 planted issues over 40 cases; the 95% intervals above are wide, and differences of a few catches are within noise.
- **One author.** All answer keys were written by one person; what counts as an error reflects their judgement.
- **LLM judge for omissions.** Completeness matches depend on the judge; every one is listed above for manual checking.
- **No live fact-checking.** Critics, adjudicator and baseline all judge facts from model knowledge only.
- **Prices** are list prices (DeepSeek peak rates; Gemini paid-tier rates although the free tier was used).
