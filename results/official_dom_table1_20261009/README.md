Official DynamicVLA DOM manipulation evaluation

Released hzxie/dynamic-vla-DOM checkpoint; no fine-tuning or retraining. The official 90 test configurations are unchanged: 20 trials each, 1800 total. Requested GPU: L40S. Pilot: one trial from each of nine dimensions; full run starts only after validation.

Results: full/00_CR through full/08_DR contain trial_results.jsonl, verification.json, logs and videos. Final table1_comparison.csv, table1_comparison.md, table1_comparison.png and summary.json appear here after all trials verify.

Success is actual manipulation completion. Time includes failed trials, using simulation time. Reference: Table 1, arXiv 2601.22153v2: 47.06% overall success, 8.53 s mean time. L40S hardware differs from the paper RTX A6000; streaming remains enabled. See protocol.json for exact revisions and provenance.

Jobs: pilot 16868046, full array 16868047, final comparison 16868048.
