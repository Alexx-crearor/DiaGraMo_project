# Giới hạn bắt buộc đi kèm kết quả

- Target DiaGraMo là `group`, một handwriting-disability grouping rộng. Không đồng nhất group1 với dysgraphia-specific flag hoặc chẩn đoán được xác nhận độc lập từng người; ongoing assessment nằm trong mô tả nguồn.
- Cohort258 giữ nguyên115/143, năm disagreement với specific flag và ba omission flags. Không đổi label/cohort trong cleanup.
- City–label association mạnh không tự chứng minh ID trực tiếp vào features hoặc thiết bị gây ra kết quả. Source effects và motor effects chưa identifiable từ dữ liệu hiện có.
- Participant-independent CV không thành source-independent CV. Historical researcher selection vẫn giới hạn các study đã inspected cohort.
- Attribution, attention, subgroup analysis và OOF bootstrap không xác lập cơ chế bệnh hay causal biomarker.
- Locked six folds hoàn tất; original full nested Phase4/SSL không được coi hoàn tất chỉ vì Lite xong. Bootstrap giữ mô hình cố định, không bao gồm toàn bộ training/selection/source uncertainty.
- Slovak120 là cohort/target/task khác. B0–B4 là internal exploratory CV tại cohort đó, không frozen external validation của DiaGraMo.
- B2 có28 order-dependent difference summaries; B4 shuffle vẫn có thể giữ vị trí ngầm. B3/B4 hard-prediction equality không chứng minh mọi temporal information vô ích.
- Không dùng individual clinical/SHAP/OOF hoặc raw trace hình nhận dạng được trong slide công khai. Hình aggregate cần giữ chú thích scope/uncertainty; audit nguồn15cities có nhóm nhỏ, trình bày nội bộ ở dạng đã gộp là ưu tiên.
