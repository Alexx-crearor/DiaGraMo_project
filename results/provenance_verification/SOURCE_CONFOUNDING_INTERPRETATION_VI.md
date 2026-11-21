# Diễn giải source–label confounding
Nguồn chính: [data descriptor](https://doi.org/10.1038/s41597-026-08625-x), [Zenodo v2](https://zenodo.org/records/21236910). Kiểm tra 2026-11-21T16:59:34.309386+00:00. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.


| hypothesis | claim | evidence_supporting | evidence_against_or_limit | required_verification | verification_status | identifiability | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | Different case/control recruitment | Published schools/PPCC referral routes; V3 city association | Not a city-level causal decomposition | Site recruitment denominators and balanced within-site sampling | SUPPORTED | Not identified causally in current cohort | S01;S13 |
| H2 | Different clinical ascertainment | Routine expert diagnosis without fixed standardized cut-off; ongoing cases included | No per-site reliability data | Blinded adjudication and shared diagnostic rubric | SUPPORTED | Not identified causally in current cohort | S01;S13 |
| H3 | Device/protocol source signal | V3 pressure/angle associations; session records absent | Paper describes common model; equal~200 median cadence across cities | Serial/driver/calibration logs and controlled cross-device recordings | UNVERIFIED | Not identified causally in current cohort | S01;S13 |
| H4 | Demographic/educational differences | Grade/gender targeted sampling; source metadata overlap | Current sample cannot separate all pathways | Site×class×grade/age overlap and recruitment design | SUPPORTED | Not identified causally in current cohort | S01;S13 |
| H5 | Preprocessing/cohort strengthens association | 14 task selection and no omit filtering verified locally | Association already observed in broader V3 selection summaries; not solely a new artifact | Original cleaning code and selection graph across source cohorts | SUPPORTED | Not identified causally in current cohort | S01;S13 |
| H6 | Multiple factors coexist | H1/H2/H4 compatible; source and outcome largely collinear | No identifiable percentage attribution | Crossed multi-site/device balanced cohort | SUPPORTED | Not identified causally in current cohort | S01;S13 |

**VERIFIED (V3):** source-only Train-fold diagnostic accuracy0.972868, CramérV≈0.946827. Metadata source không được đưa trực tiếp vào Kinematic LightGBM. Mức dự đoán cao của city không chứng minh LightGBM dùng ID/city hay direct label leakage. Nó chỉ chứng minh một đường dự đoán dataset label từ source rất mạnh trong thiết kế hiện tại.

**SUPPORTED:** tài liệu mới làm H1 và H2 có nền tảng rõ hơn; recruitment category và routine diagnostic practice cần được kiểm soát. **UNVERIFIED:** bao nhiêu phần hiệu năng LightGBM/Transformer do source, motor signal, nhân khẩu hay tác động kết hợp. Bằng chứng không đủ quy kết thành phố là thiết bị hoặc cơ sở lâm sàng.

Participant-independent6-fold đo phân biệt người mới trong hỗn hợp nguồn tương tự đã thấy, không đo chuyển sang source mới. V3 `source_fold_overlap.csv` cho mọi Test city đều đã xuất hiện ở Train. City1 có hai lớp nên tính discrimination trong nguồn được, nhưng chỉ7case: mỗi lỗi làm sensitivity đổi1/7≈14.3điểm%; không phải tập xác nhận mạnh. LOCO city1 khiến Train không có lớp0; city khác khiến Test không có lớp0. Một vài split theo school/site đủ hai lớp về đại số vẫn chưa giải quyết population/device overlap hoặc power.

Nguồn ngoài phải có cả case/control trong nhiều source, đánh giá độc lập, device/operator/batch metadata và support cho từng giao điểm; không chỉ lấy thêm nhiều samples từ source độc nhất của từng lớp. Không đổi threshold hoặc chọn architecture bằng kết quả các phân tích này.
