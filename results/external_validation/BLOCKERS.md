# Blockers

1. **Exact SVC export schema/units UNVERIFIED:**7numericcolumns have no header. c2clock/c3pen/c4-c5angles/c6pressure are supported candidates, not confirmed exporter conventions. Do not use guessed /1000 time scaling or altitude/tilt swaps. Originalrepo issue2 has no author answer.
2. **Task contract incompatible:**8conditions in paper vs14canonical DiaGraMo tasks. Actual SVCs have hw00001/one extra hw00002, no task boundaries. No exactsharedtask mapping; do not split equal-length blocks or infer tasks frompageposition.
3. **One zero-sample SVC and two negative candidate time deltas:** all120participants retain another valid recording; no source repair/exclusion applied. Need intendedsession/recording policy before signaltraining.
4. **Source/target transportability:** per-person site/device/batch absent. Case clinical-centre vs controlschool recruitment documented; no sourceheldout validation. Slovakdiag not automatically identical to DiaGraMogroup including ongoingassessment.
5. **Release restrictions:** no explicit standalone datasetLICENSE found. Boundedlocaldownload was explicitly requested and publicpaper-linked; no redistribution of raw, clinical metadata or individual OOF. XMLcontains history/comments; do not publish them.

These blockers stop signal-model operations, not structural audit, label joins, priorcontrol, tests or reporting. OutcomeE applies to learning/transfer under current evidence; rawtrajectories do exist. No fabricated features, time units, task indices, zerochannels, models or metrics.
