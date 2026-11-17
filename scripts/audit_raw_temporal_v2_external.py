"""Project-scoped external cohort inventory; never manufacture transfer data."""
from raw_temporal_v2_common import *


def main():
    folder=OUT/'phase4_complementarity/external';folder.mkdir(parents=True,exist_ok=True)
    inventory=[]
    for name in ('raw_data','clean_data','processed_data','derived'):
        directory=ROOT/'data'/name
        inventory.append(dict(path=directory.relative_to(ROOT).as_posix(),exists=directory.exists(),
            immediate_subdirectories=sorted(p.name for p in directory.iterdir() if p.is_dir()) if directory.exists() else []))
    # The present repository exposes the original SVC collection and derivatives;
    # no configured external manifest/access/licence exists. A directory name
    # alone is not evidence of an independently collected clinical cohort.
    external_manifests=list((ROOT/'configs').rglob('*external*.json'))
    assert not external_manifests,'An external manifest now exists: inspect its access/licence and compatibility before declaring unavailable'
    unexpected=[p.name for p in (ROOT/'data').iterdir() if p.is_dir() and p.name not in ('raw_data','clean_data','processed_data','derived')]
    unexpected_raw=[p.name for p in (ROOT/'data/raw_data').iterdir() if p.is_dir() and p.name!='handwriting_data']
    assert not unexpected and not unexpected_raw,'New candidate data directory appeared: inspect provenance/compatibility before declaring unavailable'
    record=dict(status='BLOCKED_DATA_NOT_AVAILABLE',scope='Current project paths and configuration, not a claim that no external datasets exist worldwide',
        inspected_paths=['data/','configs/','README.md','docs/'],inventory=inventory,
        independent_compatible_external_cohort_configured=False,external_metrics_created=False,
        reason='Only original handwriting collection and derived artifacts identified; no independently sourced compatible external cohort with project-provided access, license and harmonized labels')
    write(folder/'compatibility_audit.json',record)
    text='''# External evaluation — trạng thái và protocol

`EXTERNAL_VALIDATION_STATUS = BLOCKED_DATA_NOT_AVAILABLE`

Trong phạm vi project đã kiểm tra, chỉ xác định collection SVC gốc và các bản clean/processed/derived. Không có manifest cấu hình nguồn external độc lập với quyền truy cập/license và định nghĩa nhãn đã đối chiếu. Những participant khác hoặc bị loại trong cùng collection, nếu có, không tự trở thành external cohort. Đây không phải khẳng định rằng thế giới không có dataset thích hợp. Không tạo chỉ số external hay coi dữ liệu giả lập là external validation.

## Protocol cụ thể trước khi chạy

| Hạng mục | Bằng chứng cần có | Quyết định khi không khớp |
|---|---|---|
| Nguồn, access, license | Manifest nhà cung cấp, version/hash, điều kiện sử dụng thực tế | Không tải/sử dụng ngoài điều kiện được phép |
| Participant independence | IDs/cohort provenance không trùng collection source, split target theo participant | Không gọi internal split là external |
| Kênh tín hiệu | Mapping x/y/time/pen-status/pressure/orientation theo tên, missingness và sensor meanings | Không giả kênh không đo được là0; định nghĩa nhánh tương thích riêng |
| Device/coordinates | Đơn vị, calibration, origin/range, pressure/angle encoding | Không giả mm/lực hoặc scale-invariance; giữ device-unit audit |
| Sampling/timestamps | Clock units, acquisition order, duplicate/gap/negative/tiny Δt và pen boundaries | Audit derivatives/masks trước inference, không shuffle sample tùy tiện |
| Tuổi/task | Phân bố tuổi và định nghĩa từng tác vụ thực tế, common-task mapping | Giới hạn phạm vi transfer, không gán cùng ý nghĩa task chỉ vì tên gần giống |
| Nhãn/cohort lâm sàng | Định nghĩa target, tiêu chí nhóm và quy trình xác định nhãn | Không đồng nhất `group` hoặc diagnosis flags giữa dataset khi chưa có căn cứ |
| Preprocessing | So sánh transformation và cache fidelity trên source/target | Khóa source scaler/encoder trước target Test; bất kỳ adaptation chỉ fit target Train |
| Phương thức đánh giá | Source-only transfer và target-Train calibration/adaptation tách riêng | Không pretrain hoặc tune bằng target Test participants/labels |
| Kết quả | Metrics/CI theo participant, data shift, calibration, các task/kênh bị thiếu | Báo cáo negative/blocked; không suy ra cross-script invariance từ Czech→Slovak |

Sau khi đủ dữ liệu: khóa encoder/scaler/source model và toàn bộ metric/threshold; tạo manifest target theo participant và giữ Test bất khả dụng cho lựa chọn. Chạy source-only inference trước. Nếu label definitions tương thích và cần calibration, fit calibration chỉ trên target Train bằng inner participant CV, rồi chấm target Test đúng một lần cho protocol đã khóa. Bootstrap participant cho CI, ghi rõ không có cửa sổ nào của target Test trong SSL pretraining. Lưu checkpoint hash, participant split, paired predictions và báo cáo shift. Nếu label/task/channel không tương thích thì dừng transfer thay vì bịa mapping hoặc metrics.
'''
    (folder/'EXTERNAL_PROTOCOL_VI.md').write_text(text,encoding='utf-8',newline='\n')
    ledger('p4_external',phase='4',training_status='not_applicable',completion_status='blocked_data_not_available',
        reason=record['reason'],artifact_paths=folder.relative_to(ROOT).as_posix(),
        scientific_interpretation='Concrete transfer protocol supplied; no external metrics or invariance claim')
    progress=json.loads((OUT/'validation.json').read_text());progress.update(external_evaluation_status=record['status']);write(OUT/'validation.json',progress)
    print('EXTERNAL_VALIDATION_STATUS = BLOCKED_DATA_NOT_AVAILABLE',flush=True)


if __name__=='__main__':
    with gpu_worker():main()
