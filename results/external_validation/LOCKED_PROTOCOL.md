# Locked research protocol — before experiments

ConfigSHA256 `a75d198d92c0710b0ec2bd796b2618cf8d8764ad4a58309299618b38e3ab9af3`; locked before any model score. Dataset120participantIDs,57positive/63negative,5folds Stratified/shuffle/seed42; repeated recordings never split bysample. Target=ExcelDYSGRvs0; not samegroundtruth as DiaGraMogroup includingongoingcases.

**Admitted:** oneTrain-prior diagnostic via DummyClassifier(strategy=prior), fitonlyTrainclasscounts, constantX, thresholdp>0.5.5fits, nohyperparametersearch/scaler/calibration/earlystop. JSONcheckpointclasscounts suffice for exact replay; optimizer/RNG resumes N/A for closed-form estimator. This is external-dataset-only internalCV labelcontrol, not Kinematic, Temporal or zero-shottransfer.

**Not admitted:** A/B/Csignalexperiments and rawKinematic/TemporalD, because tasks/exportschema/units notverified. No newencoder is built to bypass physicalmeaning. NoGAF/CWT/SSL. No50/50fusion without pairedvalidKinematic/Temporalprobabilities. No source-onlyscore absentper-personsource; never usecase/controlrecruitment route as a sourcepredictor manufactured fromlabel.

ReportAccuracy,MacroF1,AUC,BalancedAccuracy,Sensitivity,Specificity,Brier,Logloss,confusion,ECE10. OverallOOFonlyif5/5folds uniqueandcomplete. 2000participantbootstrapresamples of fixedOOF,seed42; report siteclustering unhandled and conditionalnature. No pairedcomparison with DiaGraMo258 because differentcohort/target. No clinical/generalizationclaim fromprioronly.

Budgetforecast: firstcompletedpriorfit measures throughput; before followingfits budgetadmission uses measured duration plus conservative overhead and reportingreserve. NeuralGPUtensorallocation remains0. Resumecompletedfolds fromhash-verified JSON/counts and exactprivate splitIDs; no repeatfit.
