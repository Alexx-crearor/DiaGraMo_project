# Ba tuyến kiến trúc cần phân biệt

## Mốc trước: Kinematic và XAI

```mermaid
flowchart LR
 A[14 tác vụ viết] --> B[74 thống kê mỗi tác vụ]
 B --> C[1036 features]
 C --> D[LightGBM]
 D --> E[TreeSHAP và task/group analysis]
```

Code: `src/diagramo/features.py`, `branches/kinematic.py`, `kinematic_xai.py`. XAI diễn giải mô hình, không nhận diện nguyên nhân bệnh.

## Raw Temporal đã triển khai

```mermaid
flowchart LR
 A[SVC bảy tín hiệu gốc] --> B[21 processed channels]
 B --> C[Non-overlap windows tối đa128]
 C --> D[MultiScaleLocal kernels5/13/29]
 D --> E[Bốn local tokens và segment aggregation]
 E --> F[RelativeStrokeBlock hai heads]
 F --> G[Task identity14 và learned participant pooling]
 G --> H[Temporal probability pT]
 K[Kinematic LightGBM pK] --> J[Fixed50: 0.5pK + 0.5pT]
 H --> J
```

Nguồn: `src/diagramo/temporal/preprocessing.py`, `dataset.py`, `segmentation.py`, `src/diagramo/temporal_v2/multiscale.py`, config `configs/raw_temporal_v2/phase4_lite.json`. Khoảng25.252 tham số. Relative attention hoạt động giữa segment/stroke representations trong task groups; task identity embedding và pooling **không phải task-level Transformer attention**. Giữ thứ tự vật lý và mask theo code; không diễn giải attention như causal attribution.

## Slovak opaque benchmark độc lập

```mermaid
flowchart LR
 A[SVC numeric c0 đến c6] --> B[Train-only participant-balanced normalization]
 A --> S[Length và statistical LightGBM]
 B --> C[Windows256 giữ đầy đủ mẫu]
 C --> D[Ordered compact CNN B3]
 C --> E[Fixed synchronized row permutation]
 E --> F[Cùng compact CNN B4]
 D --> G[Participant OOF]
 F --> G
```

Nguồn: `scripts/native_channel_research/compact_multiscale_cnn.py`, `participant_dataset.py`, `sequence_permutation.py`. 13.761 tham số, không relative attention, task identities hoặc14task head. Đây không phải cùng Stroke Temporal25.252 tham số. Không chuyển tên numeric columns thành pressure/timestamp đã VERIFIED; B4 có thể giữ implicit positional cues qua giá trị cột.
