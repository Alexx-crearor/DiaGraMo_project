# Phase 6 — Independent Dataset Compatibility & Controlled Research

## 1. Executive Summary

**Có raw thật, nhưng chưa có điều kiện hợp lệ để so sánh Kinematic–Temporal hoặc đánh giá frozen DiaGraMo trên dữ liệu này.** Đã thực hiện tải nguồn, audit toàn bộ raw, đối chiếu nhãn, kiểm tra compatibility, reader độc lập, khóa protocol, chạy đối chứng, phân tích và kiểm thử. Không đánh dấu câu hỏi motor representation là đã được giải quyết.

Có 121 SVC: 120 recording hợp lệ và 1 file không có mẫu. Cả 120 ID vẫn có một recording hợp lệ; tổng 2,176,419 samples. Excel có 57 DYSGR và 63 nhãn 0. Bài báo mô tả tám điều kiện viết, nhưng raw không có annotation để tách các điều kiện đó. Chỉ Train-prior label control được chạy; không dùng GPU huấn luyện.

## 2. Independent Dataset Provenance

Nguồn chính là [Drotár & Dobeš (2020)](https://doi.org/10.1038/s41598-020-78611-9), Methods, Table 1/2, Supplement và [repository tác giả](https://github.com/peet292929/Dysgraphia-detection-through-machine-learning/tree/0079fdd34fc5eff0c1c58f9689127b2c73b57f8b). Commit nguồn `0079fdd34fc5eff0c1c58f9689127b2c73b57f8b` được khóa trước tải dữ liệu. Không thay thông tin tuyển mẫu bằng mô tả của paper khác.

Methods ghi nhận ba chuyên gia đánh giá và sử dụng Wacom Intuos Pro Large. Case được tuyển qua cơ sở special-needs, đối chứng qua trường học; đây là nguy cơ confounding tiếp tục cần xem xét. Supplement một trang đã được tải, render và xem trực tiếp. Hai ứng viên Pháp chỉ dùng làm bối cảnh đã kiểm tra ở Phase 5, không tải dữ liệu bị hạn chế access hoặc giả định chúng độc lập với nhau.

## 3. Actual Files and Data Schema

Ba file nguồn đã được xác minh kích thước, Git blob và SHA-256: README, Excel 17.201 bytes, ZIP 16.731.844 bytes. ZIP có 365 members/124 files: 121 SVC, hai XML và một .DS_Store; tổng 81.422.448 bytes giải nén theo inventory. Đã kiểm tra traversal, absolute paths, symlink, CRC, expansion limits và executable trước extraction. Không thực thi code hay deserialize pickle.

SVC có header đếm mẫu, sau đó là các hàng bảy giá trị số, không có tên cột. Excel có một sheet và 1.621 stored cells nhưng chỉ 120 ID có dữ liệu; các ô trống hoặc chỉ formatting không tạo thêm participant. Xem external_file_inventory.csv và external_schema_inventory.csv; đường dẫn từng recording và nội dung cá nhân chỉ giữ local.

## 4. Participant and Label Inventory

Đã join đủ 120 ID giữa Excel, filename và parent directory. Excel trộn ID dạng text năm chữ số với 49 ID dạng số ba chữ số. Quy tắc zero-padding thành năm chữ số áp dụng độc lập nhãn, không collision và khớp đủ SVC; ID nguyên gốc giữ local để đối chiếu.

Target là diag: DYSGR=1, 0=0. Một ô nhãn có công thức tham chiếu ô cùng cột: cached value đã được đối chiếu, không thực thi công thức. XML hiện tại có 192 ID và khớp 120 ID nghiên cứu; XML cũ có 155 ID, khớp 89. Không dùng các ID thử nghiệm/ngoài cohort làm subject mới. Nhãn Slovak không tự tương đương với group của DiaGraMo, vốn bao gồm ongoing assessment.

## 5. Raw Signals and Acquisition Quality

Đã khảo sát toàn bộ SVC. 120 recording có dữ liệu qua kiểm tra header/column count/finite values, gồm 2,176,419 samples. File 0 mẫu bị reader từ chối; ID tương ứng có một recording hợp lệ khác, nên không loại participant. Không ghi đè hoặc sửa file nguồn.

Cột c2 là timestamp candidate: median positive Δt=8 native units, có hai Δt âm và không có Δt=0. Có 2.430 khoảng lớn hơn mười lần median riêng recording; đây là diagnostic, không phải quy tắc sửa dữ liệu. Cột c3 binary trong các recording hợp lệ. Mọi nhãn semantic của cột và đơn vị vật lý vẫn cần xác nhận. Không sort, resample hoặc nội suy; channel summaries được tổng hợp theo participant.

## 6. Source–Label Confounding

Tài liệu mô tả tuyển case tại cơ sở special-needs và đối chứng tại trường, nên dataset thứ hai không tự giải quyết source confounding. Không có city/site/device serial/batch/date theo participant để tạo source-held-out split hoặc tính source-only negative control hợp lệ.

Không tạo source bằng cách sao chép nhãn bệnh, không suy địa điểm từ ID, không báo một perfect source control giả. Source independence và đóng góp nhân quả của nguồn vẫn UNVERIFIED. Age/sex/hand và XML comments không được dùng làm predictors.

## 7. DiaGraMo vs External Dataset Compatibility

Giữ nguyên Kinematic 14×74=1.036 và Stroke với bảy kênh gốc, 21 kênh xử lý, window tối đa 128 mẫu, kernels 5/13/29, bốn token 32 chiều, attention hai heads, task identity 14 và learned participant pooling. Không sửa kiến trúc hoặc normalization gốc.

External contract chưa có 14 task identities/instructions hay channel semantics/units được xác minh. A: NOT VALID; B: NOT FEASIBLE; C: BLOCKED; D chỉ cho phép prior label control, còn mô hình tín hiệu BLOCKED. Xem TRANSFER_PROTOCOL_DECISION.json. Dataset classification A (có raw) khác với research Outcome E (chưa có đường học tín hiệu được chấp nhận).

## 8. Task Mapping and Semantic Limitations

Đã xét đủ 112 cặp tác vụ theo nội dung, motor complexity, speed, pen lifts, duration, instruction, grade và language. Không có EXACT MATCH. Copy–sentence chỉ là CLOSE MOTOR ANALOGUE; loops–l/le chỉ WEAK ANALOGUE.

Filename hw không cung cấp annotation tám điều kiện viết. Không gán task indices bằng tên tương tự, thứ tự trên trang hoặc cắt thành tám đoạn bằng nhau. Nếu sau này có annotation, task-subset model là mô hình/protocol mới và cần development training riêng; không gọi nó là frozen model 14 tác vụ.

## 9. Channel Harmonization

external_reader.py trả đúng native c0…c6, bảo toàn thứ tự và terminal sample của recording hợp lệ; đọc lại cho kết quả giống hệt. API không nhận nhãn. File 0 mẫu bị từ chối rõ ràng.

Không có physical conversion, missing-channel imputation, fitted scaler hoặc raw-model windows. Pattern số hỗ trợ một vài candidate meanings nhưng không xác minh clock unit, coordinate scale, thứ tự góc hay altitude-to-tilt. Issue 2 trong repository gốc hỏi về bảy cột không header và chưa có câu trả lời tác giả. Generic SVC reader hoặc code thứ cấp không thay bằng chứng của exporter cụ thể. Window/mask tests ghi NOT_RUN thay vì PASS.

## 10. Locked Research Protocol

Recipe SHA-256 `a75d198d92c0710b0ec2bd796b2618cf8d8764ad4a58309299618b38e3ab9af3` được lưu trước khi tính score. Dùng participant Stratified 5-Fold, shuffle=True, seed 42; mọi recording cùng ID nằm cùng fold. Prior control dùng DummyClassifier(strategy=prior), constant X, chỉ fit tỷ lệ nhãn Train và ngưỡng cố định p>0.5.

Không hyperparameter search, scaler, early stopping, calibration fit hoặc Test-based selection. Không GAF/CWT/SSL/fusion. Khoá 2.000 bootstrap resamples của các hàng OOF theo participant; không dùng kết quả để đổi recipe. Các bước học Kinematic/Temporal không được cấp phép thực thi bởi gate semantic hiện tại.

## 11. Executed Experiments

Chỉ `Train_prior_label_control` đã fit đủ 5/5 folds. JSON checkpoint lưu class counts, private split IDs và config hash; SHA và exact replay đã kiểm tra. Không refit fold hoàn thành khi chạy resume. Với closed-form estimator, optimizer/RNG state không áp dụng; không giả báo đã kiểm tra neural resume.

Phạm vi: prior-control, 120 participants, target Excel DYSGR vs 0, participant Stratified 5-Fold/seed 42, đủ 5/5 folds. Đây là exploratory internal CV của external dataset, không phải external validation của mô hình DiaGraMo.

| model | participant_count | completed_folds | metric | value | ci95_low | ci95_high | fold_mean | fold_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train_prior_label_control | 120 | 5 | accuracy | 0.525000 | 0.433333 | 0.616667 | 0.525000 | 0.022822 |
| Train_prior_label_control | 120 | 5 | f1_macro | 0.344262 | 0.302326 | 0.381443 | 0.344144 | 0.009869 |
| Train_prior_label_control | 120 | 5 | auc | 0.479950 | 0.394880 | 0.565175 | 0.500000 | 0.000000 |
| Train_prior_label_control | 120 | 5 | balanced_accuracy | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.000000 |
| Train_prior_label_control | 120 | 5 | sensitivity | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Train_prior_label_control | 120 | 5 | specificity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Train_prior_label_control | 120 | 5 | brier | 0.249609 | 0.245068 | 0.254233 | 0.249609 | 0.001248 |
| Train_prior_label_control | 120 | 5 | log_loss | 0.692367 | 0.683275 | 0.701624 | 0.692367 | 0.002499 |
| Train_prior_label_control | 120 | 5 | ece10 | 0.000000 | 0.000174 | 0.100608 | 0.025000 | 0.005705 |

Individual OOF, split IDs và checkpoint chỉ lưu trong data đã Git-ignore. Các bảng public là aggregate. Mọi mô hình tín hiệu chưa chạy đều có lý do trong experiment_registry.csv; không dùng score 0 thay cho NOT_COMPUTED.

## 12. Kinematic vs Temporal Results

**NOT_COMPUTED.** Chưa có hai mô hình tín hiệu được đánh giá hợp lệ trên Slovak. Không so prior control Slovak với Kinematic/Temporal DiaGraMo như paired observations hoặc cùng target. Không có bằng chứng mô hình nào tốt hơn, và không suy failure mode của Temporal từ việc chưa chạy nó.

## 13. Cross-Dataset Transfer Results, if valid

Không có cross-dataset transfer hợp lệ được chạy. Không load hoặc rerun frozen checkpoints, không fit scaler mới trên external Test, không fine-tune hay điều chỉnh threshold bằng nhãn ngoài. Prior control nội bộ Slovak không phải external validation của trained DiaGraMo.

## 14. Error and Complementarity Analysis

Không có paired pK/pT nên both-correct, Kinematic-only, Temporal-only, both-wrong, corrected/harmed do fusion và probability correlation đều NOT_COMPUTED. Bảng complementarity ghi paired_participants=0 và các đại lượng chưa tính để trống, không điền số 0 giả.

Prior control có TN=63, FP=0, FN=57, TP=0 vì luôn dự đoán nhóm 0. Đây là tác động class balance của đối chứng không có tín hiệu, không phải attribution lỗi vận động. Không bootstrap chênh lệch giữa hai cohort khác nhau.

## 15. Statistical Uncertainty

CI95% là percentile từ 2.000 lần resample participant OOF với xác suất cố định. Nó không bao hàm training variance, researcher selection, source clustering hoặc domain shift. Thiếu site metadata nên không thể bootstrap source clusters; giả định IID có thể làm bất định nhỏ hơn thực tế.

Pooled prior AUC khoảng 0,47995 trong khi từng fold AUC=0,5: tỷ lệ nhãn Train thay nhẹ theo chiều ngược tỷ lệ held-out. ECE10=0 do các xác suất nằm trong cùng bin và trung bình khớp class balance, không chứng minh clinical calibration tốt. CI sensitivity/specificity suy biến là đặc tính always-predict-zero, không phải độ chắc chắn khi chuyển sang nguồn mới.

## 16. Domain Shift and Source Robustness

Không diễn giải 8 native ticks thành 8 ms hoặc 125 Hz. Không so trực tiếp các native pressure/angles/duration với DiaGraMo sau preprocessing. domain_shift_summary.csv phân biệt quan sát số, giả thuyết semantic và thông tin vật lý chưa xác minh.

Không quy khác biệt cho hardware khi chưa có cross-device hoặc session metadata. Cả hai nghiên cứu đều có nguy cơ recruitment confounding. Chưa có bằng chứng về unseen-source robustness hoặc motor biomarkers độc lập nguồn.

## 17. Compute and Reproducibility

Máy có 6 nhân vật lý/12 luồng, RAM tổng 31.75 GiB, Quadro T1000 4 GB và CUDA available. Trần session tám giờ, reserve báo cáo 15 phút, tối đa một GPU worker. Thực tế không khởi chạy GPU training worker nào.

Prior fits cộng metrics/bootstrap tốn 11.110 giây wall và 10.969 giây process CPU. RSS lớn nhất quan sát tại các checkpoint đo là 119.1 MiB; không gọi đây là OS-wide peak. GPU training và tensor allocation của thí nghiệm bằng 0. Resource query/CUDA context không phải training. Phase status JSON giữ thời gian từng bước; tổng elapsed session bao gồm đọc nguồn, viết và kiểm tra tài liệu, không nhầm với thời gian fit.

## 18. Privacy and License Restrictions

Download local có giới hạn được yêu cầu rõ trong task và nguồn nằm trong public repository được paper liên kết. Không dùng paid API/cloud GPU. Không tìm thấy standalone data LICENSE trong tree ba file; không tự áp giấy phép bài báo cho mọi nội dung XML hoặc derived clinical outputs.

Raw, participant clinical metadata, exact recording paths, splits và individual predictions đều giữ local Git-ignored. XML history/comments cần được bảo vệ. Không copy chúng sang báo cáo, không gửi email, commit/push/merge, đổi visibility hoặc rewrite history. Xem inventory và kiểm tra staging cuối cùng.

## 19. Scientific Validity and Limitations

Hoàn tất phần audit có thể thực hiện không đồng nghĩa hoàn tất so sánh representation. Grouping theo project ID được kiểm chứng; independent physical identity và source exchangeability chưa được xác nhận độc lập. Metadata/source coverage không đủ cho source-held-out claims.

Tám điều kiện trong paper, task boundaries thiếu, schema/units chưa rõ, một recording rỗng và hai candidate negative intervals đều được ghi nhận. Không có poor Temporal score để đổ cho thiếu capacity hoặc sample size: Temporal chưa chạy. Prior control không trả lời câu hỏi temporal information beyond kinematics.

## 20. Recommended Next Research Direction

Ưu tiên xin tác giả schema/exporter cho bảy cột và đơn vị, clock resets, angle conventions, ý nghĩa hw codes, task boundaries, official recording-selection policy và quyền dùng/derived release. Không gửi tự động.

Sau clarification, có thể khóa protocol D nhẹ trên Slovak hoặc một protocol task-agnostic riêng trước score, với kinematic và temporal matched inputs/budget. Chỉ mở exact frozen transfer khi full semantic contract thực sự tương thích. **CONDITIONAL GO** cho clarification và nghiên cứu phương pháp; **NO-GO hiện tại** cho signal-model comparison, clinical validation hoặc source-invariant claims. Không chạy thêm GPU chỉ vì budget còn.
