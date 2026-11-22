# Tái lập và resume

Tại project root:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/native_channel_research/run.py --max-runtime-hours 8
.venv/Scripts/python.exe -m pytest tests/native_channel_research tests/external_validation -q
```

Mỗi invocation có session budget8h, reserve15 phút. Dùng lại nguồn Phase6 đúng checksum; không tải ZIP lại. Khóa một worker; checkpoint hash/config/split/optimizer/RNG được kiểm tra, completed folds không refit. Test CPU/CUDA cần môi trường hiện tại; không claim portability tuyệt đối. Raw và OOF không public. Đổi config cần protocol/output riêng.
