"""Evidence-only integrated report and exactly one primary research direction."""
from raw_temporal_v2_common import *
from report_raw_temporal_v2_phase0 import md


def main():
    for phase in range(5):assert json.loads((OUT/f'phase{phase}_validation.json').read_text())['complete']
    p1=pd.read_csv(OUT/'tables/phase1_development_comparison.csv');p2=pd.read_csv(OUT/'tables/phase2_development_comparison.csv')
    p3=pd.read_csv(OUT/'tables/phase3_ssl_comparison.csv');comparison=pd.read_csv(OUT/'tables/phase4_nested_comparison.csv')
    delta=pd.read_csv(OUT/'tables/phase4_paired_deltas.csv');cfg=json.loads((ROOT/'configs/raw_temporal_v2/phase4.json').read_text())
    primary=comparison[comparison.seed==42].set_index('model');baseline=primary.loc['kinematic_lightgbm']
    def difference(model,reference,metric):
        rows=delta[(delta.model==model)&(delta.seed==42)&(delta.reference==reference)&(delta.metric==metric)]
        return rows.iloc[0].to_dict() if len(rows) else None
    def positive(model,reference,metric='f1_macro'):
        value=difference(model,reference,metric);return value is not None and value['lower95']>0
    conditional=[]
    for model in ('fixed_average','train_selected_weight','stacking'):
        a=difference(model,'kinematic_lightgbm','auc');b=difference(model,'kinematic_lightgbm','brier')
        if positive(model,'kinematic_lightgbm') and positive(model,'kinematic_calibrated') and a['delta']>=0 and b['delta']<=0:conditional.append(model)
    ssl_better=positive('ssl_v2','raw_v2')
    if cfg['preferred_ssl_method'].startswith('frozen_ssl_'):ssl_better=ssl_better and positive('ssl_v2','random_frozen_v2')
    paired=p2[p2.configuration.isin(['mil','stroke_transformer'])].pivot(index='seed',columns='configuration',values='f1_macro')
    temporal=bool((paired.stroke_transformer>paired.mil).all())
    selected_arch=cfg['architecture'];local_winner=p1.sort_values(['f1_macro','auc','parameters'],ascending=[False,False,True]).iloc[0]
    if not conditional:direction='E'
    elif ssl_better:direction='D'
    elif temporal and selected_arch.get('design')=='stroke_transformer':direction='C'
    elif local_winner.configuration.startswith('cnn_local_') and local_winner.configuration!='cnn_local_mean':direction='A'
    else:direction='B'
    names={'A':'Representation bottleneck','B':'Weak-supervision aggregation','C':'Temporal relational modeling',
        'D':'Self-supervised graphomotor learning','E':'Raw information is not yet demonstrably complementary'}
    raw_improved=positive('raw_v2','v1_nested_selected')
    p2mean=p2[p2.configuration.isin(['statistical','mil','stroke_transformer'])].groupby('configuration')[['f1_macro','auc']].mean()
    attention_delta=p2mean.loc['stroke_transformer']-p2mean.loc['mil']
    hypotheses=[
        dict(hypothesis='H0 aggregation',evidence=f'Best Phase1={local_winner.configuration}, development F1={local_winner.f1_macro:.6f}; RawV2 vs v1 nested paired lowerCI positive={raw_improved}',
            result='Higher aggregation is supported as a development bottleneck; complete procedure tested in nested CV',
            limitations='Single development split/multiple ablations; final change also includes optimizer/budget choices, not a pure causal intervention',decision='Retain validated aggregation reference'),
        dict(hypothesis='H1 local compression',evidence='Frozen v1 CNN pressure-std probe R2: feature-map0.407, window0.174, Train-only PCA32 control0.349; local readout predictive comparison saved',
            result='Some physical accessibility changes; local pooling not established as sole clinical bottleneck',
            limitations='544 vs32 dimensions partly controlled by PCA; probes conditional on frozen Train143 encoder and development validation reuse',decision='Keep as partial mechanism, not sole explanation'),
        dict(hypothesis='H2 temporal relations',evidence=f'Phase2 mean Transformer−MIL F1={attention_delta.f1_macro:+.6f}, AUC={attention_delta.auc:+.6f}',
            result='Development signal only; see paired seeds and relation/stress ablations',
            limitations='One development split, different parameter counts; content reversal is OOD; no full matched6fold MIL-vs-Transformer ablation',decision='Do not retain attention solely for novelty'),
        dict(hypothesis='H3 SSL',evidence=f'Preferred method={cfg["preferred_ssl_method"]}; paired SSL−Raw F1 lowerCI positive with applicable random control={ssl_better}',
            result='Pretraining benefit requires downstream controls, not reconstruction alone',
            limitations='Extra pretraining compute; frozen head/pooling differs from end-to-end; fixed72 development labels in label-count curves',decision='Retain only supported gains; preserve negative results'),
        dict(hypothesis='H4 complementarity',evidence='Qualifying prespecified conditional comparisons: '+(', '.join(conditional) or 'none'),
            result='Useful conditional value under tested fixed-fusion recipe supported' if conditional else 'Useful conditional value not robustly demonstrated under tested recipe',
            limitations='Conditional fixed-model OOF bootstrap; adaptive historical cohort; fusion base recipe differs from individually tuned Raw nested model; no external validation',
            decision='Limited fusion only; no wider fusion search in this task')]
    hypothesis_table=pd.DataFrame(hypotheses);hypothesis_table.to_csv(OUT/'tables/hypothesis_evidence.csv',index=False)
    ledger_frame=pd.read_csv(OUT/'EXPERIMENT_LEDGER.csv').fillna('')
    counts=ledger_frame.groupby(['phase','training_status','completion_status']).size().rename('ledger_rows').reset_index()
    counts.to_csv(OUT/'tables/execution_status_counts.csv',index=False)
    metrics_table=primary.loc[[m for m in ['kinematic_lightgbm','v1_nested_selected','raw_v2','ssl_v2','random_frozen_v2',
        'raw_fusion_base','fixed_average','train_selected_weight','stacking','kinematic_calibrated'] if m in primary.index]].reset_index()
    decision=dict(primary_direction=direction,name=names[direction],conditional_candidates=conditional,
        ssl_supported_against_applicable_controls=bool(ssl_better),raw_vs_v1_nested_lower_ci_positive=bool(raw_improved),
        protected_kinematic_oof=dict(accuracy=float(baseline.accuracy),f1_macro=float(baseline.f1_macro),auc=float(baseline.auc)),
        choice_scope='Research recommendation only, no predictive refit/outer-label hyperparameter selection',
        inference_scope='Exploratory internal participant CV; independent replication required')
    write(OUT/'final_decision.json',decision)
    root=f'''# DiaGraMo Raw Temporal Research Recovery — báo cáo tổng hợp

## Trạng thái và kết luận chính

Phase0–4 đã có validation riêng; notebook/delivery integrity được kiểm tra ở bước cuối, không coi code scaffold là thực nghiệm đã chạy. Protected baseline: Kinematic + LightGBM OOF Accuracy={baseline.accuracy:.6f}, F1 macro={baseline.f1_macro:.6f}, ROC-AUC={baseline.auc:.6f}. Không thay baseline hay xóa v1/checkpoints/Task1 XAI.

Khuyến nghị duy nhất: **Direction {direction} — {names[direction]}**. Đây là quyết định nghiên cứu dựa trên bằng chứng thăm dò, không phải chứng minh ưu thế lâm sàng hay triển khai mô hình. Conditional comparisons đáp ứng tiêu chí thăm dò F1 CI dương so với cả Kinematic/calibration và không giảm point AUC/Brier: `{', '.join(conditional) or 'không có'}`. Các CI không khử adaptive model-selection bias.

## Bảng so sánh cuối cùng — primary seed42

{md(metrics_table[['model','accuracy','f1_macro','auc','pr_auc','balanced_accuracy','sensitivity','specificity','brier']])}

Mean±sampleSD theo6fold, all seeds, paired participant bootstrap và computational scopes có trong `PHASE4_REPORT_VI.md` và `tables/phase4_*.csv`. `raw_fusion_base` dùng recipe cố định để cross-fit trung thực, không đánh tráo với Raw/SSL nested được tune riêng. Không kết luận Raw không có thông tin trong mọi setting nếu recipe Fusion cố định không khai thác được một standalone model mạnh hơn.

## Scientific question và evidence H0–H4

{md(hypothesis_table)}

Mục tiêu là hiểu thông tin được giữ/truy xuất ở đâu, cơ chế nào giúp dự đoán và liệu nó có thêm giá trị ngoài Kinematics.54,808,751 samples không phải54 triệu người độc lập; unit đánh giá là258 participants với group labels115/143. Mọi kết quả mới vẫn là exploratory vì cùng cohort đã được phát triển/đọc kết quả nhiều lần.

## Phase0: engineering validity và source fidelity

Audit nguồn/caches xác minh mapping participant/task/channel/timestamp, pen boundaries và valid derivatives; normalizer fit Train. CNN/TCN/hierarchical đều memorization8 và16 Train participants (F1=1, BCE<0.03);30 neural capacity fits và hai audit jobs được ghi riêng.24 synthetic tests đạt trần nên chỉ xác nhận năng lực trên fixture dễ, không giải thích hiệu năng lâm sàng.

Raw SVC và JSON merged thiếu11 sample ở3 recordings được phân biệt; baseline giữ nguyên, source mới dùng SVC như v1 helper thực sự đã chạy. Old metadata primary-input text không đúng được ghi nhận mà không sửa artifacts cũ. Clipping chủ yếu azimuth0.1719%; max float16 signed-log error0.00390625. Angle wraps và units chưa calibrated là giới hạn, không tự đổi đơn vị hoặc gán bệnh lý.

Chi tiết: `PHASE0_REPORT_VI.md`, source audit và bảo toàn hash. Tiny memorization không loại trừ overfit/underlearning trong n143/215, không chứng minh encoder học cơ chế motor đúng.

## Phase1: local information và aggregation

20 real development configurations,20 synthetic pooling fits,380 CPU probe fits. Probe physical targets thuộc đúng scope observation, được standardize Train-only; participant-balanced errors, complete-case undefined-motion rows đếm riêng. Frozen encoder được fit143 Train; inner probe CV là model selection **điều kiện trên encoder đóng băng**, không phải refit encoder ở từng inner probe split. Final72 held-out participants không fit probe/encoder, nhưng đã được dùng early stopping lịch sử, nên không gọi độc lập.

Feature-map/window dimension confounding được đối chứng PCA32 fit riêng Train. Padding/length giúp duration probe; pen-up fraction là input flag trực tiếp. R² cao không phải biological information. Mean/max/std/attention/multi-token không mặc nhiên tốt hơn; code giữ các thất bại. Optimizer cũng quan trọng: Phase1 CNN lr0.003 F1≈0.602 so với≈0.422 ở0.001, hierarchical≈0.746 so với≈0.509; duration readout winner không được diễn giải là nguyên nhân duy nhất.

`PHASE1_REPORT_VI.md` có mọi learning curve, gradient/variance, parameter/memory/time và predefined short-task diagnostics. Task-specific loader vẫn xử lý full bag, không báo timing đó như một acquisition protocol rút gọn.

## Phase2: temporal modeling

Ba họ multi-scale statistical/MIL/stroke-relative Transformer giữ backbone/mask/coverage chung, có controls time/spatial/index/original encoder và second seed cho winner Phase1. No new Task Transformer được thêm vì novelty. Hierarchical reference có task context cũ được giữ như baseline, không coi nó là ablation riêng chứng minh task attention có ích.

Transformer−MIL trên development: F1={attention_delta.f1_macro:+.6f}, AUC={attention_delta.auc:+.6f}. Đây không phải một matched6fold causal attention experiment. MIL hiện tại còn shared-score parameterization, optimizer và weak supervision là alternative explanations chưa loại trừ. Từng perturbation phân biệt metadata-consistent transport invariance với content-only inconsistent-metadata OOD stress.

## Phase3: SSL và label-count sensitivity

Masked coordinates/pressure/speed che đồng thời dependent channels+halo2, loss bỏ padding/invalid derivative/pen-up-only targets. Constant/interpolation chỉ dùng Train/visible endpoints. Reconstruction không đủ chứng minh representation tốt; scratch/frozen linear/frozen nonlinear/fine-tune cùng random-frozen controls mới kiểm tra downstream. Có hai seeds và label-count curves. Curves luôn có thêm72 labeled validation và một architecture đã phát triển trước; không gọi là tổng annotation-budget efficiency.

`PHASE3_REPORT_VI.md` ghi chi phí pretraining riêng, physical MSE target space, all negative outcomes và optional contrastive baseline không lên lịch do thiếu validated augmentation policy.

## Phase4: generalization, complementarity và XAI

Original6fold seed42, Train-only inner3 selection; all scalers/PCA/selection/SSL fits không dùng evaluation participants. Main Raw inner early-stop predictions không tái sử dụng làm combiner inputs. Strict subinner cross-fitting tạo Train OOF cho weights/meta; Kinematic-only calibration kiểm chứng conditional benefit ngoài threshold/calibration. Corrected cases đơn lẻ không đủ kết luận complementary information.

Task/segment occlusion và motor IG chỉ từ outer Train checkpoints trên outer Test. Raw logit khác probability; IG giải thích F(x)−F(reference) khi clock/flags/metadata giữ cố định. Source index/time alignment, random-head/seed/reference stability, deletion controls và failed numerical cases được ghi rõ. Không suy tremor/severity/diagnosis từ peaks; model/attribution thiếu ổn định không được dùng để diễn giải hành vi thực chất. Chi tiết/figures đúng-sai ở `PHASE4_REPORT_VI.md`.

## Chi phí, provenance và giới hạn

{md(counts)}

Ledger rows không đồng nhất số pipeline fits:38 Phase1 CPU configurations gồm380 inner/final regressor fits. Neural/SSL, CPU probes/readouts, CPU classifiers/meta và reused checkpoints có budget/statistics riêng; hai CPU audit jobs cũng bị scheduler debit bảo thủ. Resource accounting ở `RESOURCE_ACCOUNTING_VI.md`; packages/hardware ở provenance/environment JSON. GPU T1000 4GB chỉ có một worker, full samples xử lý chunked, không silently truncate. Không so cache-only baseline timing với full raw extraction như cùng scope.

Thiếu external dataset độc lập/tương thích qua project:

`EXTERNAL_VALIDATION_STATUS = BLOCKED_DATA_NOT_AVAILABLE`

Protocol external cụ thể có access/license/channel/device/time/age/task/label/cohort/preprocessing gates. Không external metrics giả hoặc cross-script invariance claim. Các giới hạn còn lại gồm n258 nhỏ, adaptive historical cohort, multiple development comparisons, shared pooling/early-stopping choices, transformed-unit probes, conditional bootstrap, fixed-fusion recipe và posthoc selected XAI cases.

## Artifact đáng giữ cho paper

Giữ matched pooling/aggregation tables với negative cases, probe dimensional controls/coverage, matched M1/M2/M3 relation ablations, scratch/frozen/random/SSL contrasts, complete six-fold OOF/fold/seed tables, paired conditional-value controls, raw-source-aligned correct/wrong XAI với numerical/stability caveats. Không gọi đây là foundation model hoặc cải thiện đã xác nhận độc lập. `FINAL_DECISION_VI.md` trả lời10 câu hỏi và nêu đúng một hướng chính.

Notebook `07_raw_temporal_research_recovery.ipynb` chỉ đọc artifacts, có cell bảng số, không huấn luyện khi mở. Nguồn công cụ quy trình: `SOFTWARE_REFERENCES.md` (đã đối chiếu metadata citation); đây không phải nguồn chứng minh model quality.
'''
    (OUT/'REPORT_VI.md').write_text(root,encoding='utf-8',newline='\n')
    next_experiment={'A':'Một controlled independent replication của representation-preserving pooling so với mean, cùng encoder/head/budget, trên participant mới.',
        'B':'Một preregistered independent replication: original shared MIL so với duration-aware và separate-level MIL, cùng backbone/optimizer, không mở thêm fusion grid.',
        'C':'Một matched independent stroke-relative-attention so với MIL experiment, với time/spatial/order controls và parameter/compute budgets khóa trước target Test.',
        'D':'Một independent SSL-versus-random/matched-scratch experiment với cùng pooling/readout và supervised budget; báo thêm total-compute cost.',
        'E':'Một independent source-to-target replication sau khi có manifest/license/channel/task/label compatibility; giữ Kinematic và Raw recipe cố định, kiểm tra paired conditional value trước khi mở lại fusion search.'}[direction]
    final=f'''# Final decision — Raw Temporal Research Recovery

**Direction {direction} — {names[direction]}** là hướng chính duy nhất. Đây là research recommendation, không đổi predictive pipeline sau outer Test.

1. **Nguyên nhân v1 có khả năng nhất:** tổng hợp/optimization/generalization ở các tầng cao là ứng viên có bằng chứng development; không thể quy duy nhất cho local mean pooling. RawV2 cải thiện rõ so với v1 nested theo paired F1 lowerCI: `{raw_improved}`; nếu không, tín hiệu development không được coi là đã tái lập.
2. **Bằng chứng trực tiếp:** tiny8/16 đều memorization; source/masks/gradients hợp lệ; local probes/ablation, optimizer controls và per-stage aggregation cùng chỉ ra nhiều cơ chế. Winner development `{local_winner.configuration}` F1={local_winner.f1_macro:.6f}; kết quả sáu folds thực tế ở bảng dưới, không thay bằng development score.
3. **Alternative explanations chưa loại trừ:** limited labels, shared scoring, objective/threshold/early-stopping, source/unit/circular-angle encoding, pooling/head capacity, cohort/device/task alignment và adaptive bias; không có external confirmation.
4. **Local pooling có chịu trách nhiệm?** Một số physical targets khó truy xuất hơn sau pooling, nhưng dimensionality/target/accessibility không đồng nghĩa clinical signal. Local readouts không tự chứng minh downstream improvement. Không xác định nó là nguyên nhân duy nhất.
5. **Hierarchical aggregation có chịu trách nhiệm?** Có controlled development evidence cho từng tầng; xem duration/uniform/meanmax/MIL đầy đủ. Final procedure không phải pure one-factor causal experiment nên không gán toàn bộ CV delta cho aggregation.
6. **Temporal Self-Attention có giúp?** Development mean Transformer−MIL F1={attention_delta.f1_macro:+.6f}, AUC={attention_delta.auc:+.6f}, chỉ một split/hai seeds và bounded controls. Không giữ Transformer chỉ vì mới, không suy causal temporal learning từ OOD reversal.
7. **SSL có giúp?** Preferred `{cfg['preferred_ssl_method']}`; criterion lower paired F1 CI dương so với Raw và random-frozen khi áp dụng: `{ssl_better}`. Reconstruction tốt không tự đủ; xem đầy đủ negative controls và total-compute caveat.
8. **Có thông tin hữu ích ngoài Kinematics?** Prespecified conditional comparisons vượt cả protected Kinematic/calibration theo tiêu chí thăm dò: `{', '.join(conditional) or 'chưa chứng minh'}`. Không suy absence of information trong mọi recipe từ kết quả âm tính của fixed-fusion feasibility, không coi corrected cases riêng là bằng chứng.
9. **Architecture giữ cho iteration tiếp:** giữ protected Kinematic làm anchor và archive đúng Raw architecture/config đã kiểm chứng: `{json.dumps(selected_arch,ensure_ascii=False,sort_keys=True)}`. Giữ artifact không có nghĩa khuyến nghị deployment; không mở rộng kiến trúc/fusion dựa vào outer score.
10. **Đúng thí nghiệm tiếp theo:** {next_experiment} External hiện blocked, cần giải quyết protocol/access trước; không tự chạy bước mới trong nhiệm vụ này.

{md(metrics_table[['model','accuracy','f1_macro','auc','brier']])}

Mọi kết luận là exploratory internal participant CV. Direction được chọn theo bằng chứng/độ hữu ích, không theo độ “mới” hay thuận tiện viết paper. Source/results/checkpoints cũ giữ nguyên; chi tiết provenance, skipped/blocked diagnostics, notebook và delivery validation ở các artifact đi kèm.
'''
    (OUT/'FINAL_DECISION_VI.md').write_text(final,encoding='utf-8',newline='\n')
    print('INTEGRATED REPORT WRITTEN; direction',direction,'DELIVERY VALIDATION STILL REQUIRED',flush=True)


if __name__=='__main__':main()
