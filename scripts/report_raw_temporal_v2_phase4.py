"""Vietnamese final-stage comparison with conditional-value controls and limits."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_temporal_v2_common import *
from report_raw_temporal_v2_phase0 import md


def main():
    assert json.loads((OUT/'phase4_validation.json').read_text())['complete']
    cfg=json.loads((ROOT/'configs/raw_temporal_v2/phase4.json').read_text())
    table=pd.read_csv(OUT/'tables/phase4_nested_comparison.csv');delta=pd.read_csv(OUT/'tables/phase4_paired_deltas.csv')
    overlap=pd.read_csv(OUT/'tables/phase4_complementarity.csv');calibration=pd.read_csv(OUT/'tables/phase4_calibration.csv')
    primary=table[table.seed==42].copy();newnames=['kinematic_lightgbm','raw_v2','ssl_v2','random_frozen_v2',
        'raw_fusion_base','fixed_average','train_selected_weight','stacking','kinematic_calibrated']
    comparison=primary[primary.model.isin(newnames)].copy()
    comparison['fold_f1_mean_sd']=comparison.apply(lambda r:f'{r.mean_f1_macro:.4f} ± {r.sd_f1_macro:.4f}',axis=1)
    weights=[]
    for fold in range(6):
        item=json.loads((OUT/'phase4_complementarity'/f'fusion_outer_{fold}/combiner.json').read_text())
        stack=json.loads((OUT/'phase4_complementarity'/f'p4_meta_outer{fold}_final_stacking/result.json').read_text())
        weights.append(dict(fold=fold,raw_weight=item['raw_weight'],kinematic_weight=1-item['raw_weight'],
            stacking_kinematic_standardized_logit_coefficient=stack['coefficient'][0][0],
            stacking_raw_standardized_logit_coefficient=stack['coefficient'][0][1],stacking_intercept=stack['intercept'][0],
            note='Probability-logit combiner coefficients, not sensor or biological attribution'))
    weights=pd.DataFrame(weights);weights.to_csv(OUT/'tables/phase4_fusion_weights.csv',index=False)
    figures=[];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig,sources,caption):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        figures.append(dict(name=name,caption=caption,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            outputs={e:sha(OUT/'figures'/(name+'.'+e)) for e in ('png','svg')}));plt.close(fig)
    a=delta[(delta.seed==42)&(delta.reference=='kinematic_lightgbm')&(delta.metric=='f1_macro')&delta.model.isin(newnames)].copy()
    fig,ax=plt.subplots(figsize=(10,5));positions=np.arange(len(a))
    ax.errorbar(a.delta*100,positions,xerr=np.array([(a.delta-a.lower95)*100,(a.upper95-a.delta)*100]),fmt='o',capsize=3)
    ax.axvline(0,color='black',linestyle='--');ax.set_yticks(positions,a.model);ax.invert_yaxis()
    ax.set_xlabel('Macro F1 difference vs Kinematic, percentage points; paired95% participant bootstrap')
    save('p4_paired_f1',fig,[OUT/'tables/phase4_paired_deltas.csv'],'OOF participant resampling with fitted models fixed; intervals do not remove historical selection bias or refit uncertainty.')
    models=[m for m in ('raw_v2','ssl_v2','raw_fusion_base') if m in overlap.model.unique()]
    fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4))
    for ax,model in zip(axes,models):
        r=overlap[(overlap.model==model)&(overlap.seed==42)&(overlap.subgroup_field=='all')].iloc[0]
        matrix=np.array([[r.both_correct,r.kin_only],[r.raw_only,r.both_wrong]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=258);ax.set_xticks([0,1],['Raw correct','Raw wrong']);ax.set_yticks([0,1],['Kin correct','Kin wrong']);ax.set_title(model)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center',color='white' if matrix[i,j]>129 else 'black')
    save('p4_error_overlap',fig,[OUT/'tables/phase4_complementarity.csv'],'Matched258 participant OOF predictions. Correcting some Kinematic errors alone is not evidence of conditional value.')
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],linestyle='--',color='black',label='Ideal reference')
    for model in ['kinematic_lightgbm','raw_v2','ssl_v2','stacking','kinematic_calibrated']:
        group=calibration[(calibration.model==model)&(calibration.seed==42)].sort_values('bin')
        x=group.mean_probability.to_numpy();v=group.positive_fraction.to_numpy();ax.plot(x,v,marker='o',label=model,markersize=4)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted group1 probability, fixed0.1-wide bins',ylabel='Observed group1 fraction');ax.legend(fontsize=7)
    save('p4_calibration',fig,[OUT/'tables/phase4_calibration.csv'],'Empirical OOF calibration,258 participants; empty bins remain gaps and all bin counts are retained in CSV. Not external clinical calibration.')
    changes=pd.read_csv(OUT/'tables/phase4_participant_changes.csv');fig,ax=plt.subplots(figsize=(7,5))
    chosen=changes[(changes.model=='stacking')&(changes.seed==42)]
    for category,marker,color in [('raw_only','o','#287A57'),('kin_only','x','#B24B3C'),('both_correct','.', '#555555'),('both_wrong','+', '#6F5194')]:
        group=chosen[chosen.category==category];ax.scatter(group.kinematic_probability,group.delta,label=f'{category} n={len(group)}',marker=marker,color=color,s=20)
    ax.axhline(0,color='black',linewidth=.6);ax.set(xlim=(0,1),xlabel='Protected Kinematic OOF probability',ylabel='Stacking probability minus Kinematic');ax.legend(fontsize=7)
    save('p4_fusion_corrected_harmed',fig,[OUT/'tables/phase4_participant_changes.csv'],'Corrected/harmed analysis is descriptive held-out error analysis, never used to alter weights, gates or thresholds.')
    write(OUT/'figures/phase4_manifest.json',figures)
    active=json.loads((OUT/'phase4_complementarity/xai/active_run.json').read_text());run=ROOT/active['run']
    xai=json.loads((run/'validation.json').read_text());cases=pd.read_csv(run/'selected_cases.csv')
    wave=[]
    for category in ('raw_only','kin_only','both_wrong','both_correct'):
        selected=cases[cases.category==category]
        if len(selected):
            pid=selected.iloc[0].participant_id;path=run/'figures'/('wave_'+pid+'.png')
            wave.append(f'### {category}: {pid}\n\n![Held-out case]({path.relative_to(OUT).as_posix()})')
    plan=json.loads((OUT/'phase4_complementarity/fusion_resource_plan.json').read_text())
    paired=delta[(delta.seed==42)&delta.reference.isin(['kinematic_lightgbm','kinematic_calibrated'])&delta.model.isin(newnames)&delta.metric.isin(['f1_macro','auc','brier'])]
    text=f'''# Phase 4 — Nested CV, complementarity và XAI

## Protocol đã khóa

258 participants, nhãn `group`, outer Stratified6fold seed42 giữ nguyên assignment. Inner3fold chỉ trong outer Train chọn learning rate hoặc readout hyperparameters và epoch budget. Encoder SSL luôn fit riêng trên participant Train tương ứng; epoch pretraining được khóa từ development trước outer evaluation. Final neural initialization seeds42/43 dùng cùng lựa chọn hyperparameter từ inner seed42; đây là sensitivity theo initialization, không phải hai cohort độc lập.

Toàn bộ thiết kế vẫn **thăm dò nội bộ** do cohort/outer outcomes lịch sử đã ảnh hưởng nghiên cứu. Chống direct fit leakage không xóa được adaptive selection bias. Kinematic baseline, Task1 XAI và v1 source/checkpoints/results giữ nguyên.

## Kết quả participant OOF và fold variability

{md(comparison[['model','accuracy','f1_macro','auc','pr_auc','balanced_accuracy','brier','fold_f1_mean_sd']])}

Bảng đầy đủ Sensitivity, Specificity,8 metrics OOF, mean±sampleSD6fold và tất cả seeds/v1 ở `tables/phase4_nested_comparison.csv`. Primary seed42 không được thay bằng seed tốt hơn sau chấm Test. Fold SD không phải CI; pooled OOF AUC và mean fold AUC khác nhau do ranking/calibration giữa folds.

![Paired F1](figures/p4_paired_f1.png)

Paired95% CIs resample258 participant OOF rows cùng chỉ số cho mọi phương pháp, giữ nguyên model đã fit; không resample windows/strokes, không refit CV, không khử adaptive-selection uncertainty. Chênh lệch là đơn vị F1/AUC/Brier; hình F1 đổi sang percentage points.

{md(paired[['model','reference','metric','delta','lower95','upper95']])}

Nếu random-frozen control xuất hiện, nó có cùng local architecture/statistical pooling và inner head search như SSL, nhưng encoder giữ initialization ngẫu nhiên. Điều này tách lợi ích encoder pretraining khỏi lợi ích pooling/readout; không gọi một pipeline mang tên SSL là bằng chứng pretraining tốt hơn nếu đối chứng không ủng hộ.

## Minimal Fusion và conditional predictive value

Raw cho Fusion dùng architecture đã khóa, recipe/hyperparameters/epochs cố định từ development. Nó được báo riêng là `raw_fusion_base`, không đánh tráo với `raw_v2`/`ssl_v2` đã chọn hyperparameters và epoch bằng inner CV. Các dự đoán inner dùng early stopping không được tái sử dụng như strict cross-fitted inputs cho combiner.

Trong mỗi outer Train215, từng inner Train143/144 có subinner3fold để tạo base OOF cho weight/stacking; combiner chỉ fit những OOF Train inputs rồi chấm inner validation72/71. Final combiner fit outer-Train OOF215, final base learners fit215, Test43 chỉ chấm một lần cho protocol đã khóa. Weight chọn F1/AUC trong Train OOF, tie ưu tiên Kinematic. Stacking là Logistic Regression C0.1 trên Train-scaled clipped logit probabilities. Calibration chỉ dùng Kinematic là đối chứng riêng: stacking hơn calibration mới hỗ trợ vai trò Raw vượt việc dịch ngưỡng/calibration đơn thuần.

{md(weights)}

![Error overlap](figures/p4_error_overlap.png)

![Corrected and harmed](figures/p4_fusion_corrected_harmed.png)

Corrected cases của một classifier yếu vẫn có thể xuất hiện. Chỉ các so sánh held-out conditional controls và uncertainty mới hỗ trợ giá trị bổ sung; không kết luận từ số `raw_only` riêng lẻ. Các subgroup đủn dùng metadata `group`/`diagnosis_dysgraphia` thực tế, chỉ mô tả error counts; không đồng nhất diagnosis flag với target `group`, không bịa demographics.

Fusion seed43 sensitivity đã chạy: `{plan['seed43_sensitivity']}`. Khi có, chỉ thay initialization của final Raw base, giữ combiner Train bằng seed42; không gọi là một meta-pipeline được chọn lại độc lập. Nếu không, budget đã dự trữ mọi primary fit và không đủ toàn bộ sáu optional fits; standalone Raw/SSL vẫn có hai seeds.

![Calibration](figures/p4_calibration.png)

## Held-out explainability và source alignment

Task occlusion của mọi procedure đã định trước dùng đúng checkpoint fit outer Train và outer Test tương ứng. Detailed target `{active['target']}` là lựa chọn **posthoc để mô tả mô hình quan sát mạnh nhất**, không refit/chọn lại predictive architecture, threshold, feature hoặc hyperparameter. Các case chọn theo cả bốn trạng thái đúng/sai với Kinematic, thiên về high-confidence theo quy tắc đã khóa nên không đại diện phân bố dân số.

IG tính trên **raw positive-group logit**, tổng bằng F(x)−F(reference) trong motor-input contrast; clock, validity flags, hierarchy metadata giữ cố định. Không gọi tổng này là xác suất hoặc toàn bộ quyết định. Motor channels là transformed/signed-log/normalized coordinates, pressure, orientation, valid velocity/acceleration; references không phải người khỏe mạnh hay counterfactual vật lý hợp lệ. Time/duration được kiểm tra thêm bằng whole-task/segment occlusion và duration-readout stress khi áp dụng.

Kiểm tra IG không đạt được giữ riêng và không dùng cho substantive interpretation: **{xai['ig_checks_failed']} reference-case checks**. Task/seed/fold stability, head randomization, reference sensitivity và deletion controls nằm trong active XAI run. High random-head similarity hoặc instability hạn chế ý nghĩa hành vi ngay cả khi completeness đạt. Attention maps không được coi là lời giải thích đầy đủ.

Nguồn index/window/segment được đối chiếu SVC: timestamp thực, raw index half-open intervals, sample coverage đúng một lần. Hình dùng source units, không giả mm/lực. Speed invalid là gap, không vẽ thành0; thinning chỉ để hiển thị và giữ mọi boundary/extrema, không bỏ sample khỏi mô hình. Các technical segments gồm pen-state runs và invalid-Δt boundaries, không mặc định đều là pen-down strokes.

{chr(10).join(wave)}

Không suy ra tremor, severity, motor impairment hay chẩn đoán từ attribution peaks. Quality gates chỉ là heuristic diagnostics, không clinical validation. Kinematic XAI gốc được giữ nguyên.

## External evaluation và chi phí

`EXTERNAL_VALIDATION_STATUS = BLOCKED_DATA_NOT_AVAILABLE`

Audit/protocol cụ thể: `phase4_complementarity/external/EXTERNAL_PROTOCOL_VI.md`. Không có external metrics, không giả cross-language label equivalence hoặc cross-script invariance. Trước transfer phải xác minh license/access, channels, device/time units, tuổi, task, labels, source/target cohort và preprocessing.

Neural fits, SSL pretraining, CPU probes/readouts, CPU classifiers/meta và checkpoint reuse được thống kê riêng trong ledger và resource-budget files. CPU head time không bao gồm encoder feature extraction; cached baseline load/inference không được gọi là thời gian huấn luyện baseline. Mọi chi phí chỉ có ý nghĩa trong scope được ghi, không so cache-only Kinematic classifier với toàn bộ raw preprocessing như cùng một công việc.

Validation: `phase4_validation.json`, `model_validation.json`, checkpoint replay CSV, crossfit provenance và active XAI validation. Báo cáo tổng hợp cùng quyết định nghiên cứu ở `REPORT_VI.md` và `FINAL_DECISION_VI.md`; notebook chỉ đọc completed artifacts. Công cụ quy trình ghi ở `SOFTWARE_REFERENCES.md`.
'''
    (OUT/'PHASE4_REPORT_VI.md').write_text(text,encoding='utf-8',newline='\n')
    progress=json.loads((OUT/'validation.json').read_text());progress.update(phase4='complete',complete=False,unfinished=['Integrated delivery validation and executed notebook']);write(OUT/'validation.json',progress)


if __name__=='__main__':main()
