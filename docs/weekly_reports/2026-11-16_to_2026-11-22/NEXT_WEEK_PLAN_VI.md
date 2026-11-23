# Kế hoạch tuần tới — theo giá trị khoa học và tính khả thi

| Ưu tiên | Công việc đề xuất | Câu hỏi / tiêu chí hoàn thành | Giới hạn |
|---|---|---|---|
| P0 | Source-aware replication/data design | Có cả hai lớp ở nhiều nguồn, metadata acquisition và prospective held-out source | Cần owner/đối tác dữ liệu; chưa có quyền hoặc nguồn mới tự động |
| P1 | Khóa shortcut hypothesis trên development cohort | Marginal-only baseline riêng; c2/channel ablation; order controls không để giá trị cột tiết lộ vị trí | Không tune lại outer OOF đã inspected; chưa chạy |
| P2 | Kiểm tra robustness/calibration có cross-fitting | Phân biệt capacity, calibration và sequential information theo budget cố định | Protocol mới, Train-only mọi fit; chưa thực hiện |
| P3 | Làm rõ exporter/clinical codebook và release permissions | Nguồn primary về units/task cutpoints/clinical rules | Không gửi liên hệ tự động; không chặn nghiên cứu opaque hợp lệ |

Quyết định: CONDITIONAL GO cho kiểm chứng giả thuyết có kiểm soát; NO-GO cho clinical biomarker/source-invariance claims. Chưa có lý do đủ mạnh để ưu tiên Transformer lớn hơn, full SSL hay full nested architecture search. Điểm94% trên Slovak không giải quyết source confounding.
