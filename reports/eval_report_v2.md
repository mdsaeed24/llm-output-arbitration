# Evaluation: multi-critic arbitration vs a single reviewer

> Generated 2026-10-21 08:42 UTC from run `reports/runs/20261021-083839` · 40 cases (10 clean, 30 with 35 hand-planted issues) · total spend $0.301 · code `ce3febb` · logic critic `gemini-3.5-flash-lite`

> **Note:** SECOND ATTEMPT, tuned on the same 40 cases after seeing v1 (eval_report.md): tighter completeness scope (critic + baseline), no-duplicate rule for the completeness critic, adjudicator told to dismiss duplicates and nitpicks. Because the changes were chosen after looking at these cases, these numbers are optimistic; v1 is the pre-registered result.

## Headline

**The multi-critic system caught 94% (33/35) of planted errors vs 97% (34/35) for a single reviewer, with 2 false flags vs 5 (0.05 vs 0.12 per case).**

- **Recall:** the full system did **not** beat the single reviewer (paired exact McNemar p = 1.000; system-only catches 0, baseline-only catches 1). With n = 35 planted issues this difference is **not** statistically significant at α = 0.05.
- **False flags:** the system raised fewer false flags (2 vs 5; ≥ medium severity: 1 vs 2). Precision 94% vs 88%.
- **Sensitivity:** 0 of the system's false flags are omission-style complaints anchored on a planted error's span (mostly the completeness critic restating a planted error). Counting those as duplicates instead: 2 vs 4 false flags, precision 94% vs 90% — the false-flag comparison does not change direction.
- **Verdicts:** pass/fail accuracy 95% vs 98% (paired p = 1.000). Clean outputs passed: 10/10 vs 10/10; flawed outputs failed: 28/30 vs 29/30.
- **Cost & latency:** $0.0057 vs $0.0017 per case (3.3× the cost) and 11.4s vs 5.0s critical-path LLM time per case.

## Results

Every row except the single reviewer is computed from the SAME system run (ablations re-use the critics' stored outputs), so they see identical critic findings.

| Variant | Recall [95% CI] | Factual | Logic | Completeness | Precision | False flags (≥ med) | Clean cases flagged | Verdict accuracy | $ / case | s / case |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full system** (3 critics + adjudicator) | 94% (33/35) [81%–98%] | 87% (13/15) | 100% (10/10) | 100% (10/10) | 94% (33/35) | 2 (1) | 0/10 | 95% (38/40) [83%–99%] | 0.0057 | 11.4 |
| **Single reviewer** (deepseek-v4-pro) | 97% (34/35) [85%–99%] | 93% (14/15) | 100% (10/10) | 100% (10/10) | 88% (35/40) | 5 (2) | 0/10 | 98% (39/40) [87%–100%] | 0.0017 | 5.0 |
| Ablation: union of critics, no adjudicator | 94% (33/35) [81%–98%] | 87% (13/15) | 100% (10/10) | 100% (10/10) | 69% (33/48) | 15 (8) | 0/10 | 95% (38/40) [83%–99%] | 0.0032 | 5.9 |
| Ablation: factual critic alone (deepseek-v4-pro) | 63% (22/35) [46%–77%] | 87% (13/15) | 90% (9/10) | 0% (0/10) | 85% (22/26) | 4 (1) | 0/10 | 72% (29/40) [57%–84%] | 0.0017 | 5.9 |
| Ablation: logic critic alone (gemini-3.5-flash-lite) | 54% (19/35) [38%–70%] | 60% (9/15) | 100% (10/10) | 0% (0/10) | 95% (19/20) | 1 (1) | 0/10 | 72% (29/40) [57%–84%] | 0.0010 | 2.2 |
| Ablation: completeness critic alone (deepseek-flash) | 29% (10/35) [16%–45%] | 0% (0/15) | 0% (0/10) | 100% (10/10) | 48% (10/21) | 11 (7) | 0/10 | 60% (24/40) [45%–74%] | 0.0005 | 2.3 |

*Recall per type uses the PLANTED type. False flags = findings that match no planted issue (every finding on a clean case is one). s / case = critical-path LLM time (slowest critic + adjudicator), excluding queueing and retry waits.*

## Sensitivity: completeness restatements of planted errors

The strict rule (agreed before the run) never lets a completeness finding match a factual/logic planted issue. A completeness critic that restates a planted error as an omission ("the summary reverses the core distinction…") is therefore a false flag under the strict rule, although it is really a duplicate. This table shows the same results with those restatements — omission findings whose quote overlaps a planted factual/logic span — set aside. It is a mechanical proxy (an anchored complaint could occasionally be about something else — check the false-flag lists) and is NOT the headline.

| Variant | False flags (strict) | Restatements of planted errors | False flags excl. restatements | Precision (strict) | Precision excl. restatements |
|---|---|---|---|---|---|
| system | 2 | 0 | 2 | 94% | 94% |
| baseline | 5 | 1 | 4 | 88% | 90% |
| union | 15 | 6 | 9 | 69% | 79% |
| critic:factual | 4 | 0 | 4 | 85% | 85% |
| critic:logic | 1 | 0 | 1 | 95% | 95% |
| critic:completeness | 11 | 6 | 5 | 48% | 67% |

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
| critic:factual | 22 | 0 |
| critic:logic | 19 | 0 |
| critic:completeness | 0 | 10 |

Judge decisions: 20 complaint(s) matched, 12 rejected · judge cost $0.0023 (not charged to any variant).

### Judge matches — spot-check these

| Case | Planted omission | Complaint | Judge's reasoning | Raised by |
|---|---|---|---|---|
| `amoxicillin-alcohol-missed` | A direct answer about alcohol: moderate drinking has no known direct interaction with amo… | The output never answers the actual question about whether alcohol can be drunk while taking amoxicillin; the closest s… | This complaint explicitly identifies the missing direct answer about alcohol and amoxicillin: that alcohol is not stric… | system, union, completeness |
| `amoxicillin-alcohol-missed` | A direct answer about alcohol: moderate drinking has no known direct interaction with amo… | The output does not answer the user's actual question about whether alcohol can be consumed while taking amoxicillin. I… | This complaint also points to the absence of an answer to the alcohol-with-amoxicillin question, which is the same miss… | baseline |
| `api-word-limit-ignored` | Compliance with the explicit 'under 50 words' limit; the answer runs to well over 100 wor… | The output ignores the explicit 'under 50 words' constraint; the full answer is roughly 100 words, about twice the limi… | The complaint explicitly identifies the failure to comply with the 'under 50 words' limit, matching the answer-key omis… | system, union, completeness |
| `api-word-limit-ignored` | Compliance with the explicit 'under 50 words' limit; the answer runs to well over 100 wor… | The output exceeds the 50-word limit specified in the original question. The entire response is approximately 100 words… | This complaint also directly points to exceeding the 50-word limit, the same omission as m1. | baseline |
| `process-thread-mixed` | An advantage of processes (e.g. isolation: a crash or memory corruption in one process do… | The question asks for one advantage of each, but only an advantage of threads is given; no advantage of processes (e.g.… | The complaint states that no advantage of processes (such as isolation or independent memory space) is given, which is … | system, union, completeness |
| `process-thread-mixed` | An advantage of processes (e.g. isolation: a crash or memory corruption in one process do… | The question asks for one advantage of each (process and thread). The output gives an advantage of threads but never gi… | This complaint likewise identifies the absence of an advantage of processes, matching the answer-key omission about pro… | baseline |
| `recursion-example-ignored` | A code example of a recursive function, which the question explicitly asked for. | The question explicitly asks to 'give a code example', but the output contains no code at all — only prose. A short fun… | The complaint states the output contains no code example despite the question asking for one, which is exactly the answ… | system, union, completeness |
| `recursion-example-ignored` | A code example of a recursive function, which the question explicitly asked for. | The original question asks for a code example, but the output provides no code example anywhere. The sentence quoted is… | This complaint also points out that no code example is provided anywhere, matching the omission of a recursive function… | baseline |
| `reverse-string-complexity-skipped` | The time complexity of the function (O(n) in the length of the string), which the questio… | The output never states the time complexity of the reversal, which the question explicitly asks for. It should say the … | The complaint says the output never states the time complexity, which is exactly the answer-key omission about the O(n)… | system, union, completeness |
| `reverse-string-complexity-skipped` | The time complexity of the function (O(n) in the length of the string), which the questio… | The question asks to state the time complexity, but the output never explicitly states it. The time complexity of both … | This complaint also points to the missing explicit statement of time complexity (O(n)), matching the same omission. | baseline |
| `student-loan-no-recommendation` | An actual recommendation or decision rule for the user's case (e.g. build an emergency fu… | The question asks 'Should I pay off... or invest?', but the output never gives a recommendation or a decision rule (e.g… | This complaint explicitly states the output never gives a recommendation or decision rule and only restates that both o… | system, union, completeness |
| `student-loan-no-recommendation` | An actual recommendation or decision rule for the user's case (e.g. build an emergency fu… | The question asks 'Should I pay off my 7% student loan or invest it in an index fund?' The output presents both sides b… | This complaint likewise says the output presents both sides but provides no recommendation or clear decision rule, leav… | baseline |
| `tcp-udp-analogy-ignored` | A real-world analogy for the TCP/UDP difference (e.g. registered mail vs postcards), whic… | The question explicitly asks for a real-world analogy, but the output gives only technical descriptions and use-case ex… | The complaint states the output lacks the requested real-world analogy and gives only technical descriptions, which is … | system, union, completeness |
| `tcp-udp-analogy-ignored` | A real-world analogy for the TCP/UDP difference (e.g. registered mail vs postcards), whic… | The question asks for an explanation using a real-world analogy, but the output provides only a technical description w… | This complaint also identifies the absence of the required real-world analogy, matching the same answer-key omission. | baseline |
| `us-government-veto-skipped` | How Congress overrides a presidential veto: a two-thirds vote in both the House and the S… | The output never explains how Congress can override a presidential veto, which is the second explicit part of the quest… | The complaint states the output fails to explain the veto override process and specifies the two-thirds vote in both ch… | system, union, completeness |
| `us-government-veto-skipped` | How Congress overrides a presidential veto: a two-thirds vote in both the House and the S… | The original question asks how Congress can override a presidential veto, but the output does not address this at all. … | This complaint also identifies the missing veto override explanation and specifies the two-thirds majority in both the … | baseline |
| `water-ethanol-why-skipped` | An explanation of why the boiling points differ: water forms a more extensive hydrogen-bo… | The question asks why the boiling points differ, but the output never explains the cause (e.g. water's stronger hydroge… | The complaint states the output never explains why the boiling points differ, citing water's stronger hydrogen bonding … | system, union, completeness |
| `water-ethanol-why-skipped` | An explanation of why the boiling points differ: water forms a more extensive hydrogen-bo… | The question asks to explain why the boiling points differ, but the output only describes practical consequences and ne… | This complaint likewise identifies the missing causal explanation, referencing weaker intermolecular forces in ethanol … | baseline |
| `ww2-end-mixed` | Any discussion of the role the atomic bombs on Hiroshima and Nagasaki (6 and 9 August 194… | The output never addresses the question's third part: the role of the atomic bombs in Japan's surrender. It discusses t… | The complaint explicitly states the output omits any discussion of the atomic bombs' role in Japan's surrender, includi… | system, union, completeness |
| `ww2-end-mixed` | Any discussion of the role the atomic bombs on Hiroshima and Nagasaki (6 and 9 August 194… | The question asks what role the atomic bombs played in Japan's surrender, but the output does not mention the atomic bo… | This complaint also identifies that the atomic bombs and their role in Japan's surrender are entirely missing, matching… | baseline |

## Critic analytics

| Critic | Model | Valid issues | Hallucinated quotes | True positives (precision) | Planted caught alone | Unique catches | Overruled by adjudicator | …of which really false |
|---|---|---|---|---|---|---|---|---|
| factual | deepseek-v4-pro | 26 | 0 (0%) | 22 (85%) | 22 | 4 | 2/26 (8%) | 2/2 |
| logic | gemini-3.5-flash-lite | 20 | 0 (0%) | 19 (95%) | 19 | 1 | 1/20 (5%) | 1/1 |
| completeness | deepseek-flash | 21 | 0 (0%) | 10 (48%) | 10 | 10 | 10/21 (48%) | 10/10 |

- **Most valid (true-positive) issues:** factual (22).
- **Overruled most often:** completeness (10/21 = 48% of its adjudicated issues).

**Agreement between critics** (pass/fail verdict each critic would give alone):

| Pair | Agree |
|---|---|
| factual – logic | 90% (36/40) |
| factual – completeness | 48% (19/40) |
| logic – completeness | 52% (21/40) |
| all critics | 45% (18/40) |

Issue groups raised by 2+ critics: 38% (18/48). Low corroboration is expected by design: each critic owns one dimension, so critics rarely flag the same span — single-source issues are what the adjudicator exists to check.

## What the adjudicator changed

- Cases adjudicated: 29; short-circuited (unanimous confident pass): 11, of which really clean: 10; fell back to union after an adjudicator error: 0.
- Issue groups ruled on: 48 (35 confirmed, 13 dismissed).
- Dismissals that removed a false flag: 13/13; dismissals that threw away a real catch: 0/13.
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
| `python3-release-date` | wrong-date, subtle | 1 | 1/1 | 1/1 | 1 | 1 | fail ✓ | fail ✓ |
| `everest-boiling-point` | wrong-number | 1 | 0/1 | 1/1 | 0 | 0 | pass ✗ | fail ✓ |
| `adult-bone-count` | wrong-number | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `mortgage-total-interest` | wrong-number, subtle | 1 | 0/1 | 0/1 | 0 | 0 | pass ✗ | pass ✗ |
| `south-pole-scott-name` | wrong-name | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `dna-photo-51-name` | wrong-name | 1 | 1/1 | 1/1 | 0 | 1 | fail ✓ | fail ✓ |
| `github-acquirer-name` | wrong-name, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `coffee-caffeine-amount` | wrong-number, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `tomato-fruit-contradiction` | contradiction | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `roth-ira-contradiction` | contradiction | 1 | 1/1 | 1/1 | 0 | 1 | fail ✓ | fail ✓ |
| `list-dict-key-contradiction` | contradiction, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `redesign-sales-unsupported` | unsupported-conclusion | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `coffee-longevity-unsupported` | unsupported-conclusion | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `active-fund-unsupported` | unsupported-conclusion, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `english-science-circular` | circular | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `unit-tests-circular` | circular, subtle | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `us-government-veto-skipped` | skipped-part | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `water-ethanol-why-skipped` | skipped-part | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `reverse-string-complexity-skipped` | skipped-part | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `student-loan-no-recommendation` | missed-point | 1 | 1/1 | 1/1 | 1 | 1 | fail ✓ | fail ✓ |
| `amoxicillin-alcohol-missed` | missed-point | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `api-word-limit-ignored` | ignored-constraint | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `recursion-example-ignored` | ignored-constraint | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `tcp-udp-analogy-ignored` | ignored-constraint | 1 | 1/1 | 1/1 | 0 | 0 | fail ✓ | fail ✓ |
| `ww2-end-mixed` | mixed, wrong-name, skipped-part | 2 | 2/2 | 2/2 | 0 | 0 | fail ✓ | fail ✓ |
| `newton-laws-mixed` | mixed, wrong-date, contradiction | 2 | 2/2 | 2/2 | 0 | 0 | fail ✓ | fail ✓ |
| `process-thread-mixed` | mixed, wrong-claim, skipped-part | 2 | 2/2 | 2/2 | 0 | 0 | fail ✓ | fail ✓ |
| `index-fund-mixed` | mixed, wrong-name, wrong-number, unsupported-conclusion | 3 | 3/3 | 3/3 | 0 | 1 | fail ✓ | fail ✓ |

## Run health

- Cases with a crashed run: 0
- Degraded verdicts (a critic or the adjudicator failed): 0
- Critic failures after retries: 0
- Failed HTTP attempts that were retried: 1

## Caveats

- **Small sample.** 35 planted issues over 40 cases; the 95% intervals above are wide, and differences of a few catches are within noise.
- **One author.** All answer keys were written by one person; what counts as an error reflects their judgement.
- **LLM judge for omissions.** Completeness matches depend on the judge; every one is listed above for manual checking.
- **No live fact-checking.** Critics, adjudicator and baseline all judge facts from model knowledge only.
- **Prices** are list prices (DeepSeek peak rates; Gemini paid-tier rates although the free tier was used).
