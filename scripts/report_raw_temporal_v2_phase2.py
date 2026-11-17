"""Report controlled temporal families using completed validated artifacts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from raw_temporal_v2_common import *
from report_raw_temporal_v2_phase0 import md


def main():
    status=json.loads((OUT/'phase2_validation.json').read_text());assert status['complete']
    config=json.loads((ROOT/'configs/raw_temporal_v2/phase2.json').read_text())
    table=pd.read_csv(OUT/'tables/phase2_development_comparison.csv');stress=pd.read_csv(OUT/'tables/phase2_perturbations.csv')
    primary=table[table.configuration.isin(['statistical','mil','stroke_transformer'])]
    grouped=primary.groupby('configuration')[['accuracy','f1_macro','auc','parameters','seconds','peak_gpu_mib']].mean().reset_index()
    means=grouped.set_index('configuration');df1=means.loc['stroke_transformer','f1_macro']-means.loc['mil','f1_macro'];dauc=means.loc['stroke_transformer','auc']-means.loc['mil','auc']
    conclusion=('Có tín hiệu development thuận lợi cho attention, nhưng hai seed/một split chưa xác nhận generalization.'
        if df1>0 and dauc>0 else 'Không thấy lợi ích nhất quán của Self-Attention so với MIL trên cả F1 macro và AUC trong đối chứng development hiện tại.')
    figures=[];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig,sources,description):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        figures.append(dict(name=name,description=description,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            outputs={e:sha(OUT/'figures'/(name+'.'+e)) for e in ('png','svg')}));plt.close(fig)
    names=list(dict.fromkeys(table.configuration));fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,metric in zip(axes,['f1_macro','auc']):
        for seed,marker in [(42,'o'),(43,'x')]:
            a=table[table.seed==seed];ax.scatter(a[metric],[names.index(n) for n in a.configuration],marker=marker,label=f'Seed {seed}')
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(metric+' (development validation n=72)');ax.legend()
    save('p2_matched_models',fig,[OUT/'tables/phase2_development_comparison.csv'],'Primary designs have two seeds; relation controls only seed42. No six-fold error bars or external validation implied.')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['content_only','high_speed']):
        a=stress[stress.stress==kind];ax.scatter(a.mean_abs_probability_change,[names.index(n) for n in a.configuration],marker='o')
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlabel('Mean absolute probability change');ax.set_title(kind+' stress, development only')
    save('p2_temporal_stress',fig,[OUT/'tables/phase2_perturbations.csv'],'Content reversal keeps original relationship slots and is deliberately metadata-inconsistent OOD. High-speed mask removes top5% motion windows, not annotated clinical events.')
    fig,ax=plt.subplots(figsize=(12,5));ax.set_axis_off();ax.set_xlim(0,12);ax.set_ylim(0,5)
    boxes=[(0,3.8,2.1,1,'21 chronological channels\nall original samples'),(2.5,3.8,3,1,'Multi-scale CNN k5/13/29\nTCN dilation2/4, dimension32'),
        (6,3.8,2.2,1,'4 retained tokens\nmasked local groups'),(8.7,3.8,3,1,'Stroke representations\ncoverage-preserving chunks'),
        (0,1.6,3.5,1.3,'M1: mean+max+std pooling\nno temporal attention'),(4.2,1.6,3.5,1.3,'M2: learned attention MIL\nno inter-stroke attention'),
        (8.4,1.6,3.5,1.3,'M3: stroke relative attention\ntime/gap/duration/spatial/pen-state'),
        (3,.1,6,1,'Task identity (unordered14 tasks) → participant pooling → binary head')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor='#EAF2F8',edgecolor='#24506B'));ax.text(x+w/2,y+h/2,label,ha='center',va='center')
    for x in [2.2,5.6,8.3]:ax.annotate('',xy=(x+.25,4.3),xytext=(x,4.3),arrowprops={'arrowstyle':'->'})
    for x in [1.75,5.95,10.15]:ax.annotate('',xy=(x,2.95),xytext=(9.8,3.7),arrowprops={'arrowstyle':'->','color':'#777777'})
    save('p2_architecture',fig,[ROOT/'configs/raw_temporal_v2/phase2.json'],'Schematic of matched controlled families; original-local MIL provides an encoder control. Approximate device coordinates are not calibrated physical distances.')
    write(OUT/'figures/phase2_manifest.json',figures)
    text=f'''# Phase 2 — Temporal modeling dưới đối chứng

Đã chạy và replay {status['development_fits']} fits.143 Train/72 validation,43 outer Test không tham gia. Thiết kế được khóa sau Phase1, hash dependency nằm trong config. Đây là nghiên cứu nội bộ thăm dò, không phải Nested CV.

## Ba họ mô hình và đối chứng

![Architecture](figures/p2_architecture.png)

Local encoder dùng kernel5/13/29, stride4, hai residual dilated convolution2/4, giữ4 token32 chiều mỗi window. M1 dùng mean/max/std, M2 dùng MIL, M3 thêm duy nhất stroke-relative attention hai heads với sequence tối đa32 stroke. Cùng backbone và cùng protocol cho phép so M2/M3; số tham số vẫn khác và được báo cáo. Đối chứng original-local MIL thay riêng backbone bằng convolution gốc, cùng4 token/aggregation. Không thêm Task Transformer vì chưa có bằng chứng biện minh.

Tất cả window được giữ. Stroke quá dài được chia nhóm bounded, các chunk gom bằng trọng số số token hợp lệ; đây không tương đương attention trên một sequence vô hạn. Task identity được dùng để phân biệt14 tác vụ, không giả làm thứ tự thời gian giữa task.

{md(table)}

![Matched models](figures/p2_matched_models.png)

Trung bình hai seed của ba thiết kế chính, không phải trung bình6fold:

{md(grouped)}

## Relative-time và spatial ablation

Time features gồm signed-log time difference, positive gap và duration ratio. Spatial là displacement giữa centroid stroke **xấp xỉ**, khôi phục từ signed-log float16 cache gốc, theo device units; không gán đơn vị mm/đặc tính sinh học. Pen-state difference và self-identity vẫn được giữ khi tắt time/spatial. Các ablation tắt riêng quan hệ, không loại các kênh input vật lý khỏi encoder; index-only dùng index cache/normalizer riêng và stroke ordinal thay thời gian vật lý.

## Perturbation và rare-event stress

{md(stress)}

Transport đảo token và metadata cùng nhau trong bounded group: kiểm tra equivariance phải không đổi dự đoán. Content-only đảo nội dung stroke nhưng giữ các slot quan hệ cũ: **metadata-inconsistent OOD stress**, không phải handwriting tự nhiên hay bằng chứng nhân quả. High-speed stress che toàn bộ window thuộc top5% speed trung bình trong các interval hợp lệ của từng participant; so với che ngẫu nhiên cùng số window. Đây là heuristic stress, không phải annotation event bệnh lý; không chọn ngưỡng từ nhãn.

![Stress tests](figures/p2_temporal_stress.png)

## Self-Attention có cung cấp thông tin hữu ích thêm không?

{conclusion} Chênh lệch trung bình Transformer−MIL: F1 macro={df1:+.6f}, AUC={dauc:+.6f}. Không suy ra kết luận nhân quả hoặc lợi ích sàng lọc chỉ từ development. Quy tắc chọn family chỉ dùng F1/AUC hai seed của ba thiết kế chính, đã ghi trước khi chấm perturbation; các đối chứng một seed không được chọn opportunistically.

Family được chọn cho Phase3: `{status['selection']['selected_design']}`; local encoder `{status['selection']['selected_local_encoder']}` theo đối chứng paired MIL seed42: chỉ chọn original khi cả F1 và AUC đều hơn multiscale. Đây là lựa chọn development một seed cho backbone, chưa chứng minh ưu thế. Phase3 phải có scratch controls khớp local encoder/family và hai seed; không foundation model, không pretrain held-out participant.

Để không bỏ một aggregation tốt chỉ vì đang thử kiến trúc mới, winner Phase1 được chạy thêm seed43 theo cùng protocol (seed42 dùng artifact đã có). Trung bình hai seed của reference Phase1: F1={status['selection']['p1_reference_two_seed_mean']['f1_macro']:.6f}, AUC={status['selection']['p1_reference_two_seed_mean']['auc']:.6f}. So sánh reference này với best primary family bằng quy tắc F1/AUC/parameter đã khóa. Chọn reference Phase1 cho SSL: `{status['selection']['phase1_reference_selected_for_ssl']}`. Local encoder thực sự dùng cho SSL: `{status['selection']['selected_local_encoder_for_ssl']}`; kiến trúc downstream đầy đủ nằm trong `development_decision.json`. Nếu không khớp một scratch configuration đã chạy thì Phase3 phải bổ sung scratch controls trước khi so fine-tuning. Reference không tham gia bảng stress M1/M2/M3; không suy rằng Transformer mới nhất là mô hình mạnh nhất.

Checkpoint replay, unit tests và transport invariance: `phase2_validation.json`. Paired predictions, learning curves và chi phí trong từng thư mục/tables. Hình là bản tổng quát cho review, chưa tuyên bố tuân thủ yêu cầu của journal cụ thể. Công cụ quy trình được ghi nhận ở `SOFTWARE_REFERENCES.md`.
'''
    (OUT/'PHASE2_REPORT_VI.md').write_text(text,encoding='utf-8',newline='\n')
    global_status=json.loads((OUT/'validation.json').read_text());global_status.update(phase2='complete',phase3='ready',complete=False);write(OUT/'validation.json',global_status)


if __name__=='__main__':main()
