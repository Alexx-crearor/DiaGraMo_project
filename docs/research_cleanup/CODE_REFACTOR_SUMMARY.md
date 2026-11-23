# Refactor nhỏ và kiểm chứng hành vi

## 1. Native CLI importability

File: `scripts/native_channel_research/run.py`. Callers: CLI `python .../run.py`, future imported `main`; internal reader/train/evaluate dependencies giữ nguyên. Trước đây `np` chỉ được import trong `__main__`, nên gọi `main()` từ module có thể lỗi NameError ở prior replay. Nay import NumPy ở module scope và nhận `main(argv=None)`; `parse_args(argv)` giữ nguyên default CLI behavior. Không đổi architecture, config, seed, threshold, aggregation, input/output paths hoặc phép tính training. Historical artifacts chỉ đọc.

Checks: CLI help stdout before/after giống nhau; tests gọi imported help/invalid args xác nhận không đọc signal hoặc gọi train. Numerical replay đọc lại fixed existing OOF, không chạy model fit. Không tuyên bố đã retrain pipeline để xác minh refactor.

## 2. Checkpoint test isolation

File: `tests/native_channel_research/test_native.py`. Hai synthetic checkpoint tests trước đây ghi `test_checkpoint.pt`/`test_cuda_exact_resume.pt` vào ignored research-local directory. Autouse fixture nay chuyển common.LOCAL và module.LOCAL sang pytest tmp_path cho từng test; không sửa source dataset, real model hoặc expected assertions. Cleanup test plugin cũng cô lập BEFORE run để không làm hỏng hiện trạng trước khi fixture mới tồn tại.

## 3. Không deduplicate khác semantics

Không gộp DiaGraMo physical preprocessing với native numeric reader; không gộp metrics/threshold/bootstrap khác contract; không đổi report.py namespaces hoặc tên checkpoint imports. Dynamic file-path imports của Phase5/6 giữ nguyên. Hardcoded Windows path duy nhất trong AST inventory cũ là fixture ZIP safety test, không production path cần thay.

## Phạm vi phép so sánh

`CLI_BEFORE.json`/`CLI_AFTER.json`, `BEHAVIOR_PRESERVATION_CHECKS.json`, `NUMERICAL_PRESERVATION.json`, `TEST_RESULTS*.json`, `PRESERVATION_VALIDATION.json`. Một historical native full-tree guard đã PASS trước cleanup; sau edits hợp lệ nó không phù hợp snapshot cũ, nên được deselect rõ ràng và thay bằng cleanup preservation gate. Không sửa historical manifest để che thay đổi.
