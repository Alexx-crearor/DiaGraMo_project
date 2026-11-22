# Phase 4 — Adapter có kiểm soát

Đã thực thi reader độc lập trên121SVC:120recording có tín hiệu trả đúng7cột/hàng theoheader và reread bit-identical; một file0mẫu bị từ chối. Không đổi thứ tự hoặc mất terminal sample của các recording hợp lệ. Không gọi hoặc sửa preprocessing DiaGraMo. API không nhận label nên label không vào array.

**PARTIAL:** chỉ hoàn thành numericadapter. Physicalharmonization, semantic taskmapping, windowing/masks, normalizer và CNNinput chưa thực hiện vì gatePhase3. Không trảzerochannels, không inferunits, không resample. Các testchưa applicable ghiNOT_RUN/NOT_APPLICABLE chứ khôngPASS. Label-onlypriorcontrol không cần chuẩn hóa.
