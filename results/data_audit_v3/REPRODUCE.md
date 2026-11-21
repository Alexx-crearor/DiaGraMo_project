# Reproduce — local read-only audit

Environment: project `.venv`, Python3.10+, NumPy/pandas/SciPy/scikit-learn/matplotlib/pytest đã cài. Không network, GPU hoặc checkpoint deserialization. Không cần cài package mới.

```powershell
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase all
```

Các pha độc lập sau initialization:

```powershell
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase abc
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase d
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase ef
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase report
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase tests
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase validate
```

Seed42; metadata smoothing2 cố định, permutation999, paired bootstrap2000, calibration10 equal-width bins. No hyperparameter/threshold search. Initial protected snapshot không ghi đè khi rerun; audit outputs riêng có thể tái tạo, original artifacts không sửa. Source metadata numeric/blank semantics được ghi; literal prefix NA được giữ.

PhaseD exhaustive4000raw files và3612cohort cached recordings; có thể tốn vài phút CPU/I/O. PROTECTED_BEFORE và final verification đọc hàng chụcGiB chỉ để hash, không copy dữ liệu. Figures PNG300dpi/SVG, participant units, all source units uncalibrated. Runtime actual trong phase status files. Manifest SHA của artifact có thể khác giữa reruns do timestamps/figure metadata; kiểm tra số liệu và source hashes, không yêu cầu byte-identical report timestamps.
