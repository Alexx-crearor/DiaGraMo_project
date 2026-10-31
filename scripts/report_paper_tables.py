"""Build the focused comparison of the 15 configurations published in Tables I–III."""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = [
    ("I — MOMENT và đặc trưng động học", [
        ("moment", "LogisticRegression", "MOMENT + Logistic Regression"),
        ("moment", "RandomForest", "MOMENT + Random Forest"),
        ("moment", "SVM", "MOMENT + SVM"),
        ("kinematic", "XGBoost", "Động học + XGBoost"),
        ("kinematic", "LightGBM", "Động học + LightGBM"),
        ("kinematic", "AdaBoost", "Động học + AdaBoost"),
    ]),
    ("II — Biểu diễn ảnh", [
        ("gaf_efficientnet_b4", "XGBoost", "GAF / EfficientNet-B4 + XGBoost"),
        ("gaf_vgg16", "RandomForest", "GAF / VGG16 + Random Forest"),
        ("gaf_vit_b_16", "LightGBM", "GAF / ViT-B16 + LightGBM"),
        ("cwt_vgg16", "SVM", "CWT / VGG16 + SVM"),
        ("cwt_vit_b_16", "SVM", "CWT / ViT-B16 + SVM"),
        ("cwt_efficientnet_b4", "LightGBM", "CWT / EfficientNet-B4 + LightGBM"),
    ]),
    ("III — Fusion (AdaBoost)", [
        ("Kinematic+MOMENT+GAF+CWT", "AdaBoost", "Động học + MOMENT + GAF + CWT"),
        ("Kinematic+GAF", "AdaBoost", "Động học + GAF"),
        ("Kinematic+MOMENT+GAF", "AdaBoost", "Động học + MOMENT + GAF"),
    ]),
]


def main():
    validation = json.loads((RESULTS / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("published_configurations_complete"):
        raise RuntimeError("The published configurations have not all completed validation.")
    with (RESULTS / "comparison_paper.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["representation"], row["classifier"]): row for row in rows}
    expected = {(rep, clf) for _, configs in TABLES for rep, clf, _ in configs}
    if len(rows) != 15 or len(lookup) != 15 or set(lookup) != expected:
        raise ValueError("Expected exactly the 15 unique published configurations.")

    lines = [
        "# Đối chiếu 15 cấu hình trong các bảng của bài báo",
        "",
        "Đã hoàn tất cả 15 cấu hình được công bố ở Bảng I, II và III. Đây là bản dựng thực nghiệm độc lập từ PDF và dữ liệu hiện có; không có mã nguồn, phụ lục hoặc cấu hình gốc của tác giả để xác nhận tái lập chính xác.",
        "",
        "Kết quả chính: động học + LightGBM đạt accuracy **84,88%**. Fusion động học + MOMENT + GAF với AdaBoost đạt **73,26%**, thấp hơn mức **85,3%** công bố **12,04 điểm phần trăm**. Vì vậy, **chưa tái hiện được kết quả fusion trọng tâm của bài báo**.",
        "",
        "## Cách đọc các bảng",
        "",
        "Mỗi ô kết quả ghi **Accuracy / F1 / AUC**, đơn vị %. Cột thực nghiệm dùng trung bình của 6 fold, F1 là **macro-F1**. Bài báo không nêu rõ cách lấy trung bình F1 nên hai giá trị F1 chỉ đối chiếu tham khảo. AUC trung bình các fold có thể khác AUC tính gộp toàn bộ dự đoán OOF; AUC gộp được giữ riêng trong CSV. Δ Accuracy = thực nghiệm − bài báo, đơn vị điểm phần trăm.",
        "",
        "Đánh giá trên 258 người (115 nhãn 0, 143 nhãn 1), chia stratified 6-fold theo người, seed 42. Mỗi fold kiểm tra có 43 người. Các bước chuẩn hóa và bộ phân loại chỉ được fit trên tập huấn luyện của fold tương ứng. Fusion dùng GAF/ViT-B16 và CWT/EfficientNet-B4 đã xác định trong quy trình.",
        "",
    ]
    ordered = []
    for table_number, (title, configs) in enumerate(TABLES, start=1):
        lines.extend([
            f"## Bảng {title}", "",
            "| Cấu hình | Bài báo: Acc / F1 / AUC | Thực nghiệm: Acc / F1 macro / AUC | Δ Accuracy |",
            "|---|---:|---:|---:|",
        ])
        for rep, clf, label in configs:
            row = lookup[(rep, clf)]
            paper = " / ".join(f"{float(row[key]):.1f}" for key in
                               ("paper_acc", "paper_f1_unspecified", "paper_auc"))
            local = " / ".join(f"{float(row[key]):.2f}" for key in
                               ("local_acc", "local_f1_macro", "local_mean_auc"))
            delta = float(row["delta_acc_pp"])
            lines.append(f"| {label} | {paper} | {local} | {delta:+.2f} |")
            ordered.append({"paper_table": table_number, "configuration": label, **row})
        lines.append("")
    lines.extend([
        "## Giới hạn ảnh hưởng đến kết luận", "",
        "- Nhãn chính là trường `group`. Có 5 người mà nhãn này khác cờ chẩn đoán dysgraphia và 3 người có cờ đề nghị loại. Các trường hợp này và phân tích độ nhạy đã được ghi trong báo cáo đầy đủ; không đổi cohort sau khi xem kết quả.",
        "- Bài báo chưa cung cấp công thức đầy đủ của 74 đặc trưng. Các công thức hiện dùng được dựng lại theo số lượng và nhóm đặc trưng đã mô tả.",
        "- Checkpoint, cách chuyển encoder sang 7 kênh, chi tiết CWT và siêu tham số bộ phân loại chưa được mô tả đủ để khôi phục duy nhất. Các lựa chọn hiện dùng, gồm AdaBoost SAMME, được ghi rõ trong quy trình.",
        "- Cùng số lượng 258 người không xác nhận cùng danh sách người và fold của tác giả. Các chênh lệch này là khả năng giải thích cần kiểm chứng, chưa phải nguyên nhân đã được chứng minh của khoảng cách kết quả.",
        "",
        "## Tệp để kiểm tra và sử dụng", "",
        "- [CSV của đúng 15 cấu hình, theo thứ tự bảng](paper_tables_comparison.csv).",
        "- [Dự đoán OOF và kết quả từng cấu hình](cv/) và [kiểm tra tính đầy đủ](validation.json).",
        "- [Báo cáo đầy đủ, biểu đồ và phân tích độ nhạy](REPORT_VI.md).",
        "- [Quy trình và toàn bộ giả định triển khai](../EXPERIMENT_PROTOCOL.md).",
        "- [Notebook đánh giá đã chạy](../notebooks/04_evaluation_and_fusion.ipynb).",
        "",
        "Nguồn giá trị công bố: PDF người dùng cung cấp, *Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis*, Bảng I–III ([bản arXiv](https://arxiv.org/abs/2607.09826v1)). Giá trị thực nghiệm được lấy từ `comparison_paper.csv`; không huấn luyện lại hay chọn lại tham số khi tạo báo cáo này.",
        "",
        "Tạo lại phần đối chiếu sau khi đánh giá: `.venv/Scripts/python.exe scripts/report_paper_tables.py`.",
        "",
    ])
    (RESULTS / "PAPER_TABLES_VI.md").write_text("\n".join(lines), encoding="utf-8")
    with (RESULTS / "paper_tables_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    print("Created results/PAPER_TABLES_VI.md and paper_tables_comparison.csv (15 configurations).")


if __name__ == "__main__":
    main()
