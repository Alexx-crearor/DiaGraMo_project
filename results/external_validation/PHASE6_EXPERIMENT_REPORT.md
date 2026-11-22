# Phase 6 — Thí nghiệm thực sự đã chạy

Chỉ chạy **Train-prior label control**,5/5folds,120uniqueIDs,participantStratified5/seed42, Trainclasscounts, fixedthresholdp>0.5. Đây là diagnosticlabelbalance trongexternaldataset, **không phải Kinematic, Temporal hoặc externaltransfer**.

OOFAccuracy=0.525000, MacroF1=0.344262, AUC=0.479950, BalancedAccuracy=0.500000, Sensitivity=0.000000, Specificity=1.000000. Các metric/CI và foldmean±SD thực nằm ở model_metrics.csv/fold_metrics.csv. AUCpriorOOF có thể khác0.5 vì Trainprior thay nhẹ ngược với tỷ lệheldoutfold; không coi là thông tin motor.

Checkpoint5folds là JSONsufficientstatistics, classcounts/splitIDs và recipehash; hashverified/replayexact; optimizer/RNGstate N/A vìclosedform. OOF và exactsplitIDs giữlocal-only. Source/time-windowmodels không fit, không loadoldcheckpoints, không dùngrawcột chưa định nghĩa. Fusion/complementarityKinematicTemporal NOT_COMPUTED. A INVALID; B NOTFEASIBLE; C vàDsignalmodels BLOCKED. **PARTIAL** đối với suite nghiên cứu, dùcontrol đã hoàn tất.

Runtimecontrol+metrics/bootstrap=11.110s wall/10.969s processCPU; observedRSSmax=119.1MiB (sampled, khôngOSpeak); GPUtraining/tensorallocation=0. BaselineCUDAcontext/resourceprobe không được gọi là GPUtraining.
