# Diagnostic attempt — not final XAI results

Lượt này qua linear additivity nhưng dừng ở fold 2 vì TreeSHAP interventional raw
lệch 6.456e-4 (probability probe lệch 7.783e-5). SHAP 0.49 independent C++ engine
lưu threshold ở float32; thay dtype input không giải quyết. Không ép base value
hoặc tăng tolerance. Lượt chính dùng lossless per-feature partition encoding cho
explainer, bảo toàn mọi split/masked coalition và model predictions; checkpoint
gốc giữ nguyên. Artifact ở đây chỉ là chẩn đoán, không dùng trong kết quả cuối.
