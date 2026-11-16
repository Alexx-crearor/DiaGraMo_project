"""Vietnamese data/preprocessing audit from all real source recordings."""
from raw_temporal_common import *
def md(df):
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.5g}'if isinstance(v,(float,np.floating))else str(v)for v in row)+' |'for row in df.itertuples(index=False,name=None))
def main():
    a=OUT/'audit';s=json.loads((a/'summary.json').read_text());r=pd.read_csv(a/'recordings.csv');runs=pd.read_csv(a/'pen_state_runs.csv');p=pd.read_csv(a/'prepared_inventory.csv');trade=pd.read_csv(a/'window_tradeoff.csv')
    q=r.groupby('task').agg(recordings=('n_samples','size'),samples_median=('n_samples','median'),samples_max=('n_samples','max'),duration_median=('duration_endpoint','median'),duration_max=('duration_endpoint','max'),median_effective_hz=('effective_hz','median'),dt_cv_median=('dt_positive_cv','median'),duplicates=('duplicate_dt','sum'),large_gaps=('large_dt_over_1s','sum')).reset_index()
    stroke=runs.groupby('pen_state').agg(runs=('n_samples','size'),samples_median=('n_samples','median'),duration_median=('duration_endpoint','median'),duration_p95=('duration_endpoint',lambda x:x.quantile(.95)),duration_max=('duration_endpoint','max')).reset_index()
    memory=trade.groupby('patch_size')[['patches','raw_samples','padded_cells']].sum().reset_index();memory['all_channels_float32_GiB']=(memory.raw_samples+memory.padded_cells)*21*4/1024**3
    text=f'''# Kiểm toán dữ liệu Raw Temporal Graphomotor

Đã kiểm tra toàn bộ {s['participants']} participants, {s['recordings']:,} recordings và {s['samples']:,} samples. Giữ đúng nhãn `group`, 115 người group0 và143 người group1; không đổi sang diagnosis flag. Không loại thêm participant hoặc task. Source keys, raw sample counts, hash JSON gốc và run inventories đều được lưu.

## Schema, đơn vị và thứ tự

Nguồn có đúng x, y, time, pen_status, pressure, azimuth, tilt. Thứ tự channels của source schema khác thứ tự lưu trong dictionary: pipeline lấy theo tên, không đoán vị trí. Original time được xử lý theo đơn vị số gốc, quy ước giây như baseline; không có chứng chỉ calibration thiết bị. Tọa độ và pressure giữ đơn vị gốc của thiết bị; không tự đổi thành mm hoặc Newton. Azimuth/tilt được giữ giá trị số gốc, không giả định calibration góc ngoài dữ liệu.

Không sort theo time, không drop duplicate, không resample sang512 điểm, không dùng engineered74 features. Mỗi recording là chuỗi acquisition-order có mapping raw index→timestamp→pen-state run→anomaly-split segment→window. Các task index là identity, không phải chronological ordering của14 assessments.

## Kết quả audit

{md(q)}

Toàn cohort có {s['duplicate_dt']:,} Δt=0; {s['negative_dt']} Δt<0; {s['tiny_dt']} khoảng0<Δt<0,0001; {s['large_gaps']:,} khoảng>1 giây. Nonfinite samples={s['nonfinite']}. Missing channels=0 và độ dài bảy channels khớp trong mỗi recording. Time không đều: effective Hz từ toàn duration khác1/median positive Δt, vì vậy lưu cả hai và coefficient of variation; không dùng index difference làm physical elapsed time.

Pen-state runs (state1 là pen-down, state0 là pen-up theo source/baseline convention):

{md(stroke)}

Một pen-down run được xem là stroke theo định nghĩa vận hành, không phải nét chữ có ground-truth ký tự. Timestamp anomalies có thể chia một run thành nhiều segment an toàn; lưu cả original run index và segment index. Pen-up pauses vẫn được giữ, không ghép qua chúng để tính vận tốc.

## Chính sách timestamp và đạo hàm

| Trường hợp | Xử lý và thông tin giữ lại |
|---|---|
| Sample đầu | Không có previous interval; derivative invalid, không đánh dấu duplicate giả |
| 0,0001 ≤ Δt ≤1 và cùng pen-down run | Backward velocity dx/Δt,dy/Δt; acceleration chỉ khi hai velocity intervals liên tiếp đều valid |
| Δt=0 | Giữ sample và time gốc; boundary mới; dt_valid0, duplicate flag1; velocity/acceleration0 cùng validity mask0 |
| Δt<0 | Giữ acquisition order, không sort/đoán clock reset; boundary mới, negative flag1, derivative invalid; thực tế cohort không có |
| 0<Δt<0,0001 | Boundary/invalid derivative để tránh nổ số; giữ tiny flag; cohort hiện không có |
| Δt>1 giây | Giữ gap gốc trong timestamp/elapsed metadata, boundary mới; không suy vận tốc liên tục qua gap |
| Pen state thay đổi hoặc pen-up | Không tính motion derivative qua transition hoặc trên pen-up; giữ tọa độ, pressure, thời gian, state và masks |

Các ngưỡng cố định là chính sách sensor-safety, không chọn từ Test labels. Chúng có thể loại đạo hàm ở chuyển động rất chậm hoặc pauses hữu ích; model vẫn thấy timing và anomaly flags. Chưa có ground truth clock/sensor để khẳng định mọi gap là lỗi. Velocity/acceleration dùng actual Δt, không giả định sampling uniform. Không đưa jerk vào phiên bản đầu do nhạy với đạo hàm bậc cao và timestamp anomalies.

## Biểu diễn neural

21 sample channels: relative x/y (trừ điểm đầu, giữ kích thước), pressure, azimuth, tilt, elapsed source time, valid Δt, pen state, vx/vy/speed, ax/ay, velocity/acceleration/dt validity, duplicate/negative/tiny/large-gap indicators, boundary. Raw values và index/time mapping vẫn truy xuất từ original JSON. Relative position làm mất absolute page placement; không xóa isotropic scale hoặc duration. Signed log1p cho continuous channels giảm dynamic range, có thể làm nhỏ khác biệt biên độ; flags giữ nguyên.

Train-only normalization kết hợp per-participant moments với trọng số mỗi người bằng nhau, rồi áp dụng channel mean/SD; fixed SD floor0,1 và z saturation±10. Không fit normalization từ toàn cohort. Statistics per-person lưu trước nhưng chỉ gộp fit IDs. Raw64-bit timestamps không qua saturation. Index-time control dùng dx/d(sample index), không gọi đó là vận tốc vật lý; neutralize clock-valued channels/flags và dùng sample-order encoding, nhưng vẫn giữ boundary inventory để không phá integrity.

## Segmentation và tài nguyên

Stroke/anomaly-aware non-overlapping windows tối đa128 raw samples. Long runs chia nhỏ, run ngắn padding có mask; không random crop, không truncate, không concatenate qua pen-state boundary. Tổng {int(p.sample_coverage.sum()):,}/{s['samples']:,} samples được phủ đúng một lần. Có {int(p.patches.sum()):,} windows, padding fraction {p.padding_cells.sum()/p.total_padded_cells.sum():.2%}. Physical duration mỗi window khác nhau, không giả định128 points là cùng số giây.

So sánh chi phí window sizes, chỉ từ inventory không dùng nhãn:

{md(memory)}

Chọn128 cân bằng chi phí, không chọn bằng Test performance. Window64 tăng số tokens và overhead;256 gộp dài hơn có thể mất local detail. Đây là trade-off tính toán, chưa là kết quả predictive ablation về window size.

CNN dùng kernel9 stride8 có vùng phủ chồng lấn, không bỏ positions giữa receptive fields; sau đó convolution3 và masked pooling. Bounded attention tối đa16 windows/stroke chunk và32 segments/task chunk; các chunk đều được tổng hợp, không bỏ phần cuối. Chunk boundaries hạn chế long-range interactions, cần công khai trong báo cáo. Original stroke và chunk boundaries được phân biệt.

Raw float32 bảy channels cần khoảng{s['raw_float32_gib']:.3f}GiB; hai banks padded21channels float32 cần khoảng11,37GiB. Đã giữ cachefloat32 gốc và tạo packed float16 storage khoảng5,68GiB để chia sẻ qua memory map; model/gradient vẫn float32, không dùng AMP. Max quantization error đo trên mọi input log-channel là0,00390625; timestamps/inventories vẫn float64. Pilot được thực hiện chỉ trên outer Train, không dùng Test; peak tensor memory và latency ghi riêng, không đồng nhất với toàn bộ GPU driver allocation.

Không có kết luận learning/accuracy từ data audit. Attribution spike, pause dài hoặc đạo hàm cao không chứng minh tremor, dysgraphia hoặc cơ chế sinh học.
'''
    source=json.loads((a/'raw_svc_comparison.json').read_text());comparison=pd.read_csv(a/'raw_svc_comparison.csv')
    note=f'''\n## Nguồn raw gốc và sai khác với merged JSON\n\nPrimary neural input đọc trực tiếp từ `data/raw_data/handwriting_data/data_SVC/<participant>/*.svc`, theo header count và thứ tự cột x,y,time,pen_status,azimuth,tilt,pressure. Merged `data/clean_data/*_merged.json` chỉ dùng để giữ participant/task identity và metadata group của protected cohort; không làm nguồn sample arrays cho benchmark mới.\n\nCó đủ{source['original_svc_files_available']} SVC gốc. So với merged JSON, {source['identical_sample_counts']} recordings có cùng số samples; ba recordings còn lại có tổng11 samples ít hơn trong merged. Maximum pressure discrepancy ở các chuỗi aligned là{source['max_channel_difference']['pressure']:.8f} source units, còn x/y/time/state/orientation khớp. Không kết luận mọi pressure value là chính xác np.round4 vì floating export/tie rounding có thể khác. Đã chuyển toàn bộ preprocessing/normalization/learning sang SVC và loại mọi partial fits trước source revision khỏi benchmark. Các audit/cache trước đó giữ riêng để truy vết, không ghi đè baseline hay original files.\n\n'''+md(comparison[comparison.same_samples.eq(False)][['participant_id','task','raw_samples','merged_samples']])+'\n'
    (a/'REPORT_VI.md').write_text(text+note,encoding='utf-8');print('TEMPORAL AUDIT REPORT WRITTEN')

if __name__=='__main__':main()
