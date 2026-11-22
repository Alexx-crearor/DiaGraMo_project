# Phase 0 — Repository, hardware và model contract

Branch `study/external-dataset-validation` tạo tại cùng commit, giữ bốn tệp untracked cũ. Không reset/stash/delete, không chạy queue cũ. Budget mặc định8giờ, reserve15phút, CLI cho phép đổi; đây là trần, không phải yêu cầu dùng hết.

CPU 6 physical/12 logical, affinity 12; RAM total 31.75GiB, available 9.80GiB; disk free 61.80GiB. CUDA available=True; GPU Quadro T1000. Framework availability chưa phải bằng chứng forward/backward; không chạy GPU nếu chưa qua data gate.

Đã kiểm tra code Kinematic, evaluate.py, preprocessing/segmentation/dataset, MultiScaleLocal/RelativeStrokeBlock/MultiScaleModel và frozen metadata. Contract chi tiết trong ORIGINAL_MODEL_CONTRACT.json. 14×74=1036; Temporal7→21channels, windows≤128, kernels5/13/29,4tokens32dims,2heads, task embedding14 và learned participant pooling. Không có task-level Transformer. Mô hình/checkpoints/OOF chỉ đọc và hash; không deserialize checkpoint để khảo sát dataset không tương thích.

Manifest bảo vệ gồm toàn tracked tree tại mốc và các model/checkpoint đã phát hiện. Đã đọc Data Audit V3, Phase5 provenance/requirements/shortlist/decision và LOCKED_RESEARCH_REPORT_VI.md. Chỉ tạo đầu ra mới dưới external_validation và dataignored.
