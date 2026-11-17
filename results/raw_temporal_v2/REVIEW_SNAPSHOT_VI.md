# Mốc tạm để thảo luận — 18/11/2026

**Nghiên cứu chưa hoàn tất.** Đây là snapshot trên nhánh `study/raw-temporal-research-v2`, không phải báo cáo kết quả cuối cùng. Các tiến trình GPU vẫn chạy; kết quả phát sinh sau snapshot sẽ được lưu ở mốc tiếp theo.

## Đã chạy và kiểm tra

- Phase 0: kiểm tra nguồn, preprocessing, masks, gradients, tiny-set memorization và synthetic fixtures; xem `PHASE0_REPORT_VI.md` và `phase0_validation.json`.
- Phase 1: 20 cấu hình development, các đối chứng pooling/aggregation và physical probes; xem `PHASE1_REPORT_VI.md` và `phase1_validation.json`.
- Bộ unit tests mới: 50/50 đạt. Notebook `notebooks/07_raw_temporal_research_recovery.ipynb` đã execute, có bảng số và biểu đồ; chỉ đọc artifacts, không tự huấn luyện.
- Phase 2: đã hoàn tất 5/12 lượt: statistical seed 42/43, MIL seed 42/43 và Stroke Transformer seed 42. Seed 43 của Stroke Transformer đang chạy. `tables/phase2_development_comparison.csv` là bảng tạm, chưa có validation toàn Phase 2.

## Những điểm cần thảo luận

Trên development split 143 Train / 72 validation, cấu hình Phase 1 `hierarchical_task_to_person_duration` đạt macro-F1 0.808784 và AUC 0.921094. Stroke Transformer seed 42 ở Phase 2 đạt macro-F1 0.746875, AUC 0.847656. Đây là kết quả một development split, không được đối chiếu trực tiếp như bằng chứng vượt Kinematic OOF sáu fold.

Các readout local mới chưa tự chứng minh cải thiện downstream. Các synthetic fixtures đạt trần nên không phân biệt được cơ chế gây yếu trên dữ liệu thật. Shared pooling, optimizer, kích thước representation và cách early stopping vẫn là những giải thích thay thế phải giữ trong phân tích.

## Chưa thực hiện xong

- Các lượt Phase 2 còn lại, replay checkpoint, stress tests và quyết định thiết kế đã khóa quy tắc lựa chọn.
- Phase 3: masked motor SSL, scratch/random controls, frozen readouts và label-count sensitivity.
- Phase 4: Nested CV sáu outer folds, inner ba folds; Fusion cross-fitted, đối chứng Kinematic calibration, OOF, paired bootstrap, XAI held-out và báo cáo tổng hợp.
- Validation bàn giao cuối cùng và kiểm tra external-data compatibility.

Code cho các bước tiếp theo đã được bổ sung, nhưng code có mặt và unit tests đạt **không có nghĩa các thực nghiệm đó đã hoàn tất**. Cấu hình dependent chỉ được tạo khi phase trước vượt validation. `validation.json` vẫn giữ `complete: false`.

Baseline, dữ liệu gốc và kết quả cũ được bảo toàn. Checkpoints/cache cục bộ không đưa lên GitHub; provenance và hash được lưu trong artifacts. Mọi suy luận từ cohort đã được nghiên cứu nhiều lần này là thăm dò nội bộ; chưa có xác nhận độc lập và chưa chứng minh cải thiện so với baseline Kinematic.
