"""Vietnamese evidence-bounded synthesis and publication-oriented local figures."""
from __future__ import annotations
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from phase_d import FEATURES
from scipy.stats import kruskal


def text(name: str,content: str) -> None:
    (OUT/name).write_text(content,encoding='utf-8')


def body(content: str) -> str:
    return content.split('\n',1)[1].replace('\n## ','\n### ')


def figures() -> None:
    target=OUT/'figures';target.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    def save(name: str,figure: Any) -> None:
        for extension in ('png','svg'):figure.savefig(target/f'{name}.{extension}',dpi=300,bbox_inches='tight')
        plt.close(figure)
    table=read('source_label_contingency.csv');city=table[table.field=='id_city'].copy()
    city=city.sort_values('source',key=lambda s:s.str.extract(r'(\d+)')[0].astype(int))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(len(city))
    ax.bar(x,city.group0,color='#0072B2',label='group 0');ax.bar(x,city.group1,bottom=city.group0,color='#D55E00',hatch='//',label='group 1')
    for at,n in zip(x,city.n):ax.text(at,n+1,str(n),ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=city.source,ylabel='Participants',title=f'Source-label composition (n={int(city.n.sum())}; counts, not causal evidence)');ax.tick_params(axis='x',rotation=45);ax.legend()
    save('source_label_composition',fig)
    people=read('sensor_quality_by_participant.csv');people=people[people.included].copy();rng=np.random.default_rng(SEED)
    features=[('estimated_rate_per_source_time','Estimated samples / source time unit'),('pressure_mean','Pressure (source units)'),('azimuth_mean','Azimuth (source units)'),('tilt_mean','Tilt (source units)')]
    cities=city.source.tolist();fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,(feature,label) in zip(axes.flat,features):
        data=[pd.to_numeric(people.loc[people.id_city==source,feature],errors='coerce').dropna().to_numpy() for source in cities]
        ax.boxplot(data,showfliers=False,widths=.6)
        for i,source in enumerate(cities):
            part=people[people.id_city==source]
            for group,color,marker in [(0,'#0072B2','o'),(1,'#D55E00','x')]:
                values=pd.to_numeric(part.loc[pd.to_numeric(part.group)==group,feature],errors='coerce').dropna()
                ax.scatter(i+1+rng.uniform(-.18,.18,len(values)),values,s=9,color=color,marker=marker,alpha=.55)
        ax.set(xticks=np.arange(1,len(cities)+1),xticklabels=cities,ylabel=label);ax.tick_params(axis='x',rotation=65);ax.grid(axis='y',alpha=.2)
        if feature=='estimated_rate_per_source_time':
            values=pd.to_numeric(people[feature]);centre=float(values.median())
            ax.set_ylim(centre*.9,centre*1.1);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.text(.02,.96,f'All approximately {centre:.6f}; floating range {np.ptp(values):.2e}',transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle(f'Participant summaries, n={len(people)}; equal task weighting; group0 circle / group1 cross\nSource differences may reflect population, execution, protocol or hardware')
    save('participant_sensor_signatures',fig)
    bins=read('calibration_bins.csv');fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,scope in zip(axes,['all_258','within_city1']):
        for model,color,marker in [('Kinematic_LightGBM','#0072B2','o'),('Stroke_Temporal','#D55E00','s'),('Fixed50','#009E73','^')]:
            part=bins[(bins.model==model)&(bins.scope==scope)].copy()
            for field in ['mean_probability','observed_rate','wilson_low','wilson_high']:part[field]=pd.to_numeric(part[field],errors='coerce')
            ax.plot(part.mean_probability,part.observed_rate,color=color,marker=marker,label=model)
            present=part.n>0;at=part[present]
            ax.errorbar(at.mean_probability,at.observed_rate,yerr=np.vstack([np.maximum(0,at.observed_rate-at.wilson_low),np.maximum(0,at.wilson_high-at.observed_rate)]),fmt='none',ecolor=color,alpha=.25)
        n=int(bins[(bins.model=='Fixed50')&(bins.scope==scope)].n.sum())
        positives=int(bins[(bins.model=='Fixed50')&(bins.scope==scope)].group1.sum())
        ax.plot([0,1],[0,1],'k--',linewidth=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted P(group=1)',ylabel='Observed group1 fraction',title=f'{scope}: n={n}, positives={positives}');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('OOF diagnostics only; 10 fixed bins; Wilson95% participant-bin intervals; no calibration fit')
    save('calibration_audit',fig)
    errors=read('error_overlap_by_source.csv');errors=errors[errors.scope=='city'].set_index('stratum').loc[cities]
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');y=np.arange(len(errors))
    ax.barh(y,errors.fusion_corrected,color='#009E73',label='Corrected');ax.barh(y,-errors.fusion_harmed,color='#D55E00',hatch='//',label='Harmed')
    ax.axvline(0,color='black',linewidth=1);ax.set(yticks=y,yticklabels=[f'{s} (n={n})' for s,n in zip(errors.index,errors.n)],xlabel='Participants: harmed (left), corrected (right)',title='Fixed50 effects relative to Kinematic; source-stratified OOF');ax.legend();save('fusion_effects_by_source',fig)


def main() -> None:
    if not all((OUT/f'{phase}_STATUS.json').exists() for phase in ['ABC','D','EF']):raise AssertionError('Required audit phase incomplete')
    inventory=read('data_inventory.csv');flow=read('cohort_selection_flow.csv');labels=read('label_consistency.csv');people=read('participants_inventory.csv');cohort=read('audit_cohort.csv')
    dictionary=read('metadata_dictionary.csv');associations=read('source_association_metrics.csv');controls=read('metadata_shortcut_metrics.csv');contingency=read('source_label_contingency.csv')
    overlap=read('source_fold_overlap.csv');feasibility=read('source_validation_feasibility.csv');duplicates=read('duplicate_metadata_ids.csv');identity=read('identity_link_anomalies.csv')
    sensor=read('sensor_quality_by_recording.csv');fidelity=read('preprocessing_fidelity.csv');acquisition=read('acquisition_signature_analysis.csv');quality_source=read('sensor_quality_by_source.csv')
    model=read('matched_model_metrics.csv');by_source=read('source_stratified_model_metrics.csv');effects=read('error_overlap_by_source.csv');calibration=read('calibration_diagnostics.csv');sensitivity=read('label_sensitivity.csv');ci=read('paired_model_uncertainty.csv')
    decisions=read('research_decisions.csv');joins=read('metadata_join_checks.csv');source_compare=read('svc_merged_comparison.csv');context=json.loads((OUT/'execution_context.json').read_text())
    city=contingency[contingency.field=='id_city'];br=city[city.source=='city1'].iloc[0];other=city[city.source!='city1']
    city_assoc=associations[associations.field=='id_city'].iloc[0]
    source_control=controls[controls.control=='id_city'].iloc[0];prior=controls[controls.control=='Train_class_prior'].iloc[0]
    macro_ci=ci[ci.metric=='f1_macro'].iloc[0];all_effect=effects[effects.scope=='all'].iloc[0]
    rawselected=sensor[sensor.selected];clinical_fields=dictionary[dictionary.domain=='clinical_or_screening']
    participant_sensor=read('sensor_quality_by_participant.csv');participant_sensor=participant_sensor[participant_sensor.included]
    precision=[]
    for field,decimals in [('median_dt_positive',9),('estimated_rate_per_source_time',6)]:
        values=pd.to_numeric(participant_sensor[field]);rounded=values.round(decimals)
        parts=[rounded[participant_sensor.id_city==source].to_numpy() for source in sorted(participant_sensor.id_city.unique())]
        statistic,p=kruskal(*parts) if np.ptp(np.concatenate(parts))>0 else (0.,1.)
        precision.append(dict(feature=field,n_participants=len(values),observed_min=float(values.min()),observed_max=float(values.max()),
            observed_range=float(np.ptp(values)),diagnostic_rounding_decimals=decimals,rounded_range=float(np.ptp(rounded)),
            rounded_city_kruskal_p=float(p),scope='Numerical sensitivity diagnostic only; raw values and original tests unchanged'))
    precision=pd.DataFrame(precision);csv('acquisition_precision_sensitivity.csv',precision)
    flagged=labels[pd.to_numeric(labels.flag_to_omit_total)==1];disagreements=labels[labels.label_disagreement]
    selector=cohort.set_index('participant_id');clinical_cross=pd.crosstab(cohort.group,cohort.diagnosis).rename_axis('group').reset_index();clinical_cross.columns=['group','diagnosis0','diagnosis1']
    csv('label_crosstab.csv',clinical_cross)
    identity_missing=int((~identity.metadata_id_present).sum())
    # Participant-level OOF has no per-task attribution. Quality associations are descriptive only.
    cases=read('fusion_corrected_harmed.csv')[['participant_id','fusion_effect']]
    task_quality=rawselected.merge(cases,on='participant_id',validate='many_to_one')
    case_rows=[]
    for (task,effect),part in task_quality.groupby(['task','fusion_effect']):
        for feature in FEATURES:case_rows.append(dict(task=task,fusion_effect=effect,feature=feature,n_participants=part.participant_id.nunique(),
            mean=float(pd.to_numeric(part[feature],errors='coerce').mean()),median=float(pd.to_numeric(part[feature],errors='coerce').median()),
            interpretation='Descriptive acquisition statistic, not task attribution or evidence that task caused correction'))
    csv('fusion_case_task_quality.csv',pd.DataFrame(case_rows))
    figures()
    phase_a=f'''# Phase A — Inventory, provenance và tái tạo cohort

**Verified fact:** audit đọc lại toàn bộ merged files và SVC theo đường dẫn thật trong code. Các con số là ID/files theo từng nguồn, không tự coi các ID khác nhau là những người vật lý khác nhau.

{md(inventory)}

{md(flow)}

Điều kiện `prepare.py`: đủ14 canonical tasks, không trùng task, arrays cùng chiều và ít nhất4 samples, không nonfinite, `group`∈{{0,1}}. `flag_to_omit_total` không tham gia eligibility. Tái tạo IDs/labels khớp `kinematic.npz` và six-fold assignments. Không tạo feature matrix mới, không đổi cohort.

Phát hiện source-link: {len(identity)} merged IDs có liên kết nguồn không đầy đủ; {identity_missing} trong số đó thiếu cognitive metadata. Không trường hợp nào thuộc cohort258. Có IDs trong CSV nhưng không có SVC và IDs có SVC nhưng không match CSV chính xác; không tự sửa mã hay gán nhãn. `identity_link_anomalies.csv` lưu phép kiểm tra chuỗi và `participant_source_links.csv` lưu liên kết thật. Không có bằng chứng đủ để gộp các ID này thành cùng người.

CSV cognitive có {len(duplicates)} ID lặp. Các khác biệt nằm ở các trường đánh giá HLS trong `duplicate_metadata_ids.csv`; không phải hai dòng hoàn toàn giống nhau. Group, cờ dysgraphia, city/site trên cohort được đối chiếu với mọi giá trị của các dòng lặp, không chọn dòng theo kết quả mô hình.

Kinematic đọc per-sample merged JSON qua `diagramo.branches.kinematic.extract` (alias `features.kinematic`), 74 features/task. Temporal hiện tại đọc **SVC gốc**, 21 channels, NPZfloat32 và packed float16; đây không phải cùng đường dữ liệu với Kinematic. Source migration và sự khác biệt export được giữ nguyên, kiểm toán lại ở Phase D. Normalization chỉ lấy statistics của Train fold.

`metadata_dictionary.csv` thống kê tất cả trường từ CSV và merged: nguồn, dtype, missingness, cardinality, domain. Các trường city, school/site, site_type, kỳ thu thập, age, sex/class/language thật sự tồn tại. Không có bằng chứng local về model thiết bị/serial, clock calibration, firmware hoặc recruitment SOP. Không suy phần cứng từ city/site.

Hash provenance của nguồn đọc nằm trong `source_provenance.json`; `PROTECTED_BEFORE.csv` bao phủ input/cache/checkpoint/code và artifacts được bảo vệ. Không copy raw signals hoặc cognitive item rows vào audit output.
'''
    phase_b=f'''# Phase B — Label validity

`group` là target thực tế. `diagnosis` trong audit là **alias của `special_care_type_dysgraphia`**, không phải một nhãn lâm sàng độc lập mới phát hiện. Dữ liệu nhị phân và quan hệ giữa chúng được xác minh; định nghĩa chẩn đoán/cách tuyển nhóm chưa có codebook độc lập để xác nhận.

{md(read('label_dictionary.csv'))}

{md(clinical_cross)}

Có {len(disagreements)} participant hai trường không khớp và {len(flagged)} participant mang omission flag. Không thể tự suy trường nào đúng lâm sàng. `label_disagreements.csv` chứa riêng các disagreement; `omission_flag_details.csv` chứa flagged participants cùng các component flags quan sát; `label_consistency.csv` là bảng đầy đủ. Kiểm tra OR của component flags với total chỉ là tính nhất quán quan sát, không thay cho codebook/quy tắc chính thức; không loại người hoặc sửa nguồn.

**Phát hiện provenance mới:** raw cognitive CSV không có `flag_to_omit_total` và các omission flags như merged. Vì vậy không được ghi “đã đối chiếu flag với CSV gốc”. `metadata_join_checks.csv` đánh dấu trường vắng mặt là UNVERIFIABLE, không cho pass giả. Processed RDS tồn tại nhưng local không có pyreadr/Rscript để giải mã; không cài/tải qua mạng. Cần xác nhận pipeline tạo flags và ý nghĩa exclusion từ chủ dữ liệu. Các component flags hiện có là quan sát, không tự coi tổng OR là quy tắc chính thức nếu thiếu code tạo cờ.

Sensitivity dưới đây chỉ dùng cùng xác suất OOF đã huấn luyện cho `group`. Các cohort253/255 và nhãn diagnosis là phân tích hậu nghiệm, không phải retraining hoặc independent validation, không dùng chọn winner.

{md(sensitivity[['model','cohort','target','n','group0','group1','accuracy','f1_macro','auc','brier']])}

Câu hỏi cần chủ dữ liệu: group định nghĩa bằng chẩn đoán hay recruitment/screening nào; dysgraphia flag do ai đánh giá, công cụ/ngưỡng và thời điểm nào; giải thích từng disagreement; nguồn tạo các omission flags và cohort paper chính thức; xử lý duplicate cognitive ID và HLS rows nào được merge.
'''
    phase_c=f'''# Phase C — Source–label confounding

{md(city)}

City1 có {int(br.group0)} group0/{int(br.group1)} group1. Các city khác có tổng {int(other.n.sum())} người, {int(other.group0.sum())} group0 và {int(other.group1.sum())} group1. Đây là **statistical association**, không phải city gây Dysgraphia.

{md(associations[['field','n','categories','mixed_sources','single_class_sources','label_purity','cramers_v','chi_square','chi_square_asymptotic_p','expected_cells_lt5','permutation_p']])}

Purity = tổng số người thuộc lớp đa số trong từng source / tổng cohort, chỉ là mô tả toàn cohort. Cramér V không bias-corrected. Chi-square dùng correction=False; nhiều expected cells<5 nên p asymptotic chỉ tham khảo. Kiểm chứng bằng999 participant-label permutations, seed42 (p tối thiểu1/1000), không phải phép kiểm định nhân quả và chưa giải quyết clustering/tuyển mẫu.

{md(overlap[['field','fold','train_sources','test_sources','unseen_test_sources','test_participants','fraction_test_source_seen']])}

`fold_source_class_distributions.csv` liệt kê source và class counts trong Train/Test của từng fold. Participant độc lập nhưng nguồn có thể đã thấy trong Train. Không gọi đây là source-independent validation.

## Negative control hợp lệ

Mapping nguồn chỉ dùng nhãn Train của năm folds khác. Train prior=(n1+1)/(n+2); source probability=(source_n1+2×Train prior)/(source_n+2). Source chưa thấy fallback Train prior; missing source là category riêng. Strength2 cố định, không tune. Không đưa participant ID/prefix vào diagnostic và không đưa source metadata vào clinical models.

{md(controls[['control','n','accuracy','f1_macro','auc','balanced_accuracy','brier']])}

Diagnostic city đạt accuracy {source_control.accuracy:.6f} so với prior {prior.accuracy:.6f}. Đây là khả năng dự đoán **nhãn dataset từ nguồn** dưới CV có source overlap, không phải clinical classifier. Không dùng purity tính toàn cohort để giả làm OOF.

## Identifiability

Không tách được hiệu ứng source và group cho các city chỉ có một lớp. Trong city1 có overlap hai lớp nên có thể mô tả/discriminate trong cùng nguồn, nhưng chỉ {int(br.group1)} positives nên không đủ để khẳng định generalization rộng hay sensitivity chính xác. Nếu giữ city1 làm Test, Train không còn lớp0; nếu giữ city khác làm Test, Test chỉ một lớp. Leave-One-City-Out không giải quyết vấn đề. Site-level có thể có split chứa hai lớp theo đại số nhưng cần xem support và city/protocol vẫn chồng lấp. Bảng đầy đủ `source_validation_feasibility.csv` không đánh đồng khả thi đại số với đủ lực thống kê.
'''
    compare_selected=source_compare[source_compare.selected]
    differences=int((~compare_selected.sample_count_equal).sum())
    sensor_br=quality_source[(quality_source.stratification=='id_city')&quality_source.feature.isin(['estimated_rate_per_source_time','pressure_mean','azimuth_mean','tilt_mean'])]
    phase_d=f'''# Phase D — Sensor, acquisition và preprocessing

Đọc SVC gốc: {len(sensor)} recordings, {sensor.participant_id.nunique()} directory IDs; cohort bảo vệ có {len(rawselected)} recordings/{rawselected.participant_id.nunique()} participant. Tổng samples cohort={int(rawselected.samples.sum())}. Header count,7 columns, finite values và pen states được kiểm tra. Bảng recording và participant được giữ riêng; không coi hàng triệu samples là hàng triệu replicates độc lập.

Sampling-rate estimate =1/median(positive Δt), đơn vị **samples/source-time-unit**, không xác nhận Hz vì không có chứng chỉ clock units. Duration và pen up/down valid duration dùng source-time; valid intervals trong[1e-4,1] theo code hiện tại. Khoảng không hợp lệ vẫn được đếm, không sort/drop/impute. Min observed step là khoảng cách nhỏ nhất giữa giá trị quan sát, không phải ADC resolution đã chứng minh.

**Negative evidence quan trọng:** participant cadence nằm trong [{pd.to_numeric(participant_sensor.estimated_rate_per_source_time).min():.12f}, {pd.to_numeric(participant_sensor.estimated_rate_per_source_time).max():.12f}], gần200 ở mọi city. Range chỉ {np.ptp(pd.to_numeric(participant_sensor.estimated_rate_per_source_time)):.3e}. Rank tests trên số float chưa làm tròn có thể cho p nhỏ dù chênh lệch chỉ ở sai số biểu diễn. `acquisition_precision_sensitivity.csv` giữ nguyên test gốc và thêm diagnostic rounding (Δt9 chữ số, rate6 chữ số); không đổi preprocessing/model. Không dùng p nhỏ này để tuyên bố sampling hardware khác nhau.

{md(precision[['feature','n_participants','observed_range','diagnostic_rounding_decimals','rounded_range','rounded_city_kruskal_p']])}

{md(sensor_br[['stratum','feature','n_participants','mean','median','p25','p75']],70)}

{md(acquisition[['feature','contrast','test','n','effect','p','fdr_bh_global']],60)}

Các test dùng **participant summaries** với equal-task weighting;16 acquisition indicators được khai báo trước trong code. Kruskal-Wallis across cities; Mann–Whitney giữa group hoặc trong city1; BH điều chỉnh trên toàn bộ các test hợp lệ. Missing-test p không được thay bằng0. Rank-biserial dương nghĩa group0 lớn hơn, không phải hướng tác động bệnh. City sizes không cân bằng, city1 chỉ ít positives, school/population clustering chưa mô hình hóa; p/q không phải bằng chứng thiết bị gây khác biệt.

Azimuth/tilt summaries ở đây là arithmetic mean/std trong conventions gốc; nếu có circular wrap, chúng không tương đương circular orientation statistics. Chưa xác minh angular units nên không tự đổi degrees/radians hoặc gán ý nghĩa sinh cơ học. Figure cadence dùng trục±10% median để không phóng đại sai số float10⁻¹⁰ thành khác biệt acquisition lớn; raw statistics và numerical-sensitivity table vẫn được giữ đầy đủ.

![Participant-level acquisition distributions](figures/participant_sensor_signatures.png)

## Fidelity toàn cohort

Raw SVC→21-channel signed-log float32 được dựng lại cho mọi recording cohort; independent dt/pen masks; window boundaries không vượt segment; coverage không mất samples. NPZ x/index channels, inventory, lengths và packed float16 được đối chiếu toàn bộ. Statistics gốc được tính lại trên float32 per participant; mean/SD của normalization lấy đúng215 Train, participant-balanced, SD floor0,1; flags không scale; clipping±10. Không refit clinical model và không load pickle/checkpoint weights.

Số recording source/merged có sample count khác nhau trong cohort: {differences}. `svc_merged_comparison.csv` còn kiểm tra exact channel hashes và pressure round4; không coi hai nhánh dùng input hoàn toàn đồng nhất khi conversion có khác biệt. Kinematic xử lý timestamp/đạo hàm khác Temporal: bỏ mẫu có local Δt≤0 trước gradient và gradient qua cả in-air; Temporal giữ mẫu, mask/reset đạo hàm theo pen/time boundaries. Đây là khác biệt triển khai đã tồn tại, không phải direct label leakage tự động.

Packed quantization/clipping theo recording/source được ghi ở `preprocessing_fidelity.csv` và `preprocessing_by_source.csv`; distortion phải xét số liệu này, không suy rằng “float16 hẳn vô hại” hoặc “preprocessing làm mất tín hiệu bệnh”. Không có phép ánh xạ causal từ clipping vào lỗi mô hình trong audit này.

**Hypotheses còn mở:** khác biệt sampling/pressure/angle có thể đến từ thiết bị, firmware, clock conventions, người thực hiện, tác vụ, population hoặc protocol. Chỉ thấy association không xác định được nguyên nhân. Cần device IDs, export/software versions, units/calibration, operator/SOP và recordings crossover hoặc hai lớp trong cùng điều kiện để phân biệt.
'''
    br_metrics=by_source[(by_source.source_field=='id_city')&(by_source.source=='city1')]
    phase_e=f'''# Phase E — OOF dưới source confounding

OOF được đọc trực tiếp từ baseline và sáu Temporal fold files, kiểm tra unique/missing IDs, labels/fold IDs, sigmoid(logit), threshold p>0,5 và pF=0,5pK+0,5pT. Checkpoint chỉ hash; audit không train lại, không thực hiện mới network inference. Replay ở đây là kiểm toán số học OOF/metrics; prior checkpoint replay evidence được giữ phân biệt. Fit-ID/normalizer-ID lists phải khớp215 Train, không giao43 Test. Tất cả8 metrics khớp reports trước trong tolerance1e-12.

{md(model[['model','n','accuracy','f1_macro','auc','balanced_accuracy','sensitivity','specificity','brier','log_loss']])}

## Within city1 (không phải external validation)

{md(br_metrics[['model','n','group0','group1','tn','fp','fn','tp','sensitivity','specificity','auc','sensitivity_wilson_low','sensitivity_wilson_high']])}

Ở nguồn một lớp, AUC và balanced accuracy là N/A; sensitivity hoặc specificity thiếu mẫu số cũng N/A. Macro-F1 báo với fixed labels[0,1] chỉ để minh bạch phép tính, không dùng so sánh source-invariance giữa nguồn thiếu một lớp. Các Wilson intervals là mô tả binomial theo participant, chưa bao gồm clustering/model selection; city1 rất ít positive nên bất định lớn.

{md(effects[effects.scope.isin(['all','city'])])}

Fusion sửa đúng {int(all_effect.fusion_corrected)} và làm sai {int(all_effect.fusion_harmed)} dự đoán Kinematic. Both-correct/only-correct/both-wrong và probability correlation có theo fold/city/site. Các case không tự chứng minh incremental motor information. Mỗi OOF participant dùng cả14 tasks; không có task-level prediction trong file OOF nên **không xác định được task nào gây correction**. `fusion_case_task_quality.csv` chỉ mô tả quality theo task/case group, không gán attribution.

{md(ci)}

Paired bootstrap2.000 lần seed42, resample participant trong từng label và giữ mô hình cố định; không giải quyết source clustering hay researcher selection bias. ΔF1 Fixed50−Kinematic={macro_ci.delta:+.6f}, CI95%[{macro_ci.lower95:+.6f},{macro_ci.upper95:+.6f}]. Không chọn threshold/model/calibration từ CI.

{md(calibration[['model','scope','n','group1','brier','log_loss','ece10']])}

![Calibration diagnostics](figures/calibration_audit.png)

Reliability dùng10 equal-width bins cố định, empty bins không điền số giả. ECE nhạy binning; phải đọc cùng Brier/log-loss và class mix. Averaging có thể thay calibration/threshold effects mà chưa chứng minh temporal signal độc lập; audit không fit temperature/Platt/isotonic. Cần train-only calibrated Kinematic control trong nghiên cứu riêng mới kiểm tra công bằng giả thuyết đó.
'''
    feasible_rows=[]
    for field,part in feasibility.groupby('field'):
        feasible_rows.append(dict(source_field=field,sources=len(part),sources_with_two_classes=int(part.heldout_has_two_classes.sum()),
            sources_allowing_both_train_test_classes=int(part.algebraically_two_class_loso.sum()),
            largest_within_source_minority=int(part.within_source_minority.max())))
    csv('validation_protocol_summary.csv',pd.DataFrame(feasible_rows))
    phase_f=f'''# Phase F — Validation feasibility và quyết định

{md(pd.DataFrame(feasible_rows))}

- **A — Original six-fold:** đánh giá khả năng dự đoán group cho người mới trong hỗn hợp nguồn hiện tại, với source overlap. Không chứng minh generalization nguồn mới, chẩn đoán lâm sàng hay model selection độc lập.
- **B — Within-source:** city1 có hai lớp nhưng minority nhỏ; với6fold chỉ1–2 positives/Test nên sensitivity nhảy50–100 điểm% theo một người. Không trình bày như nghiên cứu đủ lực thống kê. Một số school/site có hai lớp nhưng cũng rất ít minority; metadata population/protocol chưa cân bằng.
- **C — Leave-One-Source-Out:** city1 Test có hai lớp nhưng Train thiếu lớp0; city khác Test chỉ có một lớp. Site/date held-out có thể khả thi theo đại số ở vài nguồn, chưa đủ support và có thể vẫn overlap city/hardware. Không tự chạy một CV mới hoặc gọi nó là giải pháp confounding.
- **D — External validation:** cần cohort độc lập theo recruitment/site/time và decision process; reference labels có định nghĩa/công cụ/người đánh giá, omission policies,14-task compatibility, channels/timestamp units, device/firmware/export version, calibration/SOP/operator, missingness và harmonization plan Train-only được khóa trước Test. Cần cả hai lớp ở Test và fit population phù hợp.
- **E — Multi-source balanced collection:** thu cả hai lớp ở nhiều nguồn, có clinical ascertainment tương đương và điều kiện acquisition được ghi nhận; tránh source gần như đồng nhất label. Dự tính sample size từ độ chính xác mong muốn của sensitivity/specificity và paired improvement, tính clustering theo site, không chọn một con số tùy tiện. Có thể bổ sung crossover thiết bị hoặc standardized reference traces để phân biệt acquisition và population.

{md(decisions)}

**Quyết định tổng thể: CONDITIONAL GO cho nghiên cứu phương pháp thăm dò có giới hạn; NO-GO cho confirmatory/clinical/source-invariant claims trên thiết kế hiện tại.** Ưu tiên data/codebook/provenance và independent balanced validation; không tự tăng Transformer, thêm nhánh ảnh, SSL hoặc tìm fusion để chạy theo F1. Không kết luận LightGBM/Transformer đã học sai; hiện không định lượng được bao nhiêu phần performance đến từ source shortcut thay vì motor evidence.
'''
    for name,content in [('A',phase_a),('B',phase_b),('C',phase_c),('D',phase_d),('E',phase_e),('F',phase_f)]:text(f'PHASE_{name}_REPORT_VI.md',content)
    summary=f'''# Executive Summary — DiaGraMo Data Audit V3

## Kết luận chính

**CONDITIONAL GO** cho nghiên cứu phương pháp thăm dò; **NO-GO** cho tuyên bố clinical biomarkers, source-invariance hay cross-domain generalization từ cohort hiện tại. Chưa có cơ sở ưu tiên tăng độ phức tạp mô hình. Không chứng minh mô hình đã học sai; chỉ xác định các giới hạn của dữ liệu và phép đánh giá.

## Các verified facts và associations quan trọng

1. Tái tạo đúng {len(cohort)} IDs/labels/folds; {len(people)} merged files và {int(inventory.set_index('entity').loc['participants_with_svc','count'])} SVC directory IDs không phải cùng một inventory. Có {len(identity)} source-link anomalies ngoài cohort, không tự sửa danh tính.
2. Group và specific dysgraphia flag khác ở {len(disagreements)} người; {len(flagged)} người mang omit flag vẫn được giữ theo baseline. Raw CSV thiếu total omission flag; clinical label definitions/flag-generation provenance chưa xác nhận. Duplicate cognitive ID có các trường HLS xung đột.
3. City1: {int(br.group0)} group0/{int(br.group1)} group1; city khác: {int(other.group0.sum())}/{int(other.group1.sum())}. City–group V={city_assoc.cramers_v:.6f}. Mapping city chỉ học Train đạt OOF accuracy={source_control.accuracy:.6f}, prior={prior.accuracy:.6f}. Đây là diagnostic source→dataset label, không clinical classifier.
4. Source overlap làm participant-independent CV không trở thành source-independent. City-held-out không có split vừa Train vừa Test đủ hai lớp. Within-city1 chỉ ít positives nên không có precision đáng tin để khái quát rộng.
5. Raw SVC/cached21channels/statistics được kiểm tra toàn cohort; preprocessing/source differences được lượng hóa ở PhaseD, không suy hardware causation. Units thiết bị và clinical codebook chưa xác minh.
6. Full OOF giữ kết quả cũ: Kinematic tốt hơn fixed average về point Macro-F1; fusion corrected={int(all_effect.fusion_corrected)}, harmed={int(all_effect.fusion_harmed)}, ΔF1={macro_ci.delta:+.6f}, CI95%[{macro_ci.lower95:+.6f},{macro_ci.upper95:+.6f}]. Source/subgroup/sensitivity đều hậu nghiệm; không chọn winner.

{md(model[['model','accuracy','f1_macro','auc','brier']])}

## Việc cần làm trước mô hình tiếp theo

Xin tác giả/chủ dữ liệu codebook clinical, cohort list/omit rules, lý do disagreement, pipeline CSV/RDS→merged và duplicate-ID resolution; device/clock/export/operator/SOP/recruitment records. Thiết kế dữ liệu có cả hai lớp tại nhiều nguồn và test độc lập. C1 structured-temporal mechanism cần equal-capacity/equal-budget controls và OOD-aware perturbations riêng; C2 complementary motor evidence cần calibrated Kinematic control Train-only và validation độc lập. Chưa chạy thêm thí nghiệm.

Direct leakage, source confounding và researcher-level selection bias là ba vấn đề khác nhau. Audit code/fit-ID provenance không phát hiện đường trực tiếp đưa source ID/cognitive label vào features, nhưng không thể chứng minh vắng mọi leakage hay phân rã motor/source causally. Các giả thuyết population, protocol và thiết bị còn unidentifiable.

Code mới `scripts/data_audit_v3/`, tests riêng `tests/data_audit_v3/`, mọi outputs `results/data_audit_v3/`. Không thay đổi input/cohort/model/folds/checkpoints/OOF cũ; xem `VALIDATION_CHECKS.json` và manifest để biết phạm vi xác minh cuối cùng. Không push hoặc merge.
'''
    text('EXECUTIVE_SUMMARY_VI.md',summary)
    comprehensive=f'''# DiaGraMo — Comprehensive Data Validity & Source Confounding Audit V3

## 1. Executive Summary

{summary.split('## Kết luận chính',1)[1].replace(chr(10)+'## ',chr(10)+'### ')}

## 2. Dataset Inventory & Provenance

{body(phase_a)}

## 3. Cohort Selection Validity

{md(flow)}

Original eligibility được tái thực thi trên toàn bộ merged files; không áp dụng omission rules mới. `participants_inventory.csv` và `selection_by_source.csv` giữ cả người bị loại, lý do thiếu tasks/labels và source missingness. ID duy nhất là độc lập theo ID trong code, chưa phải xác nhận hồ sơ rằng mọi ID đều là người vật lý khác nhau; anomalies ngoài cohort đã được ghi riêng.

## 4. Label Definition & Consistency

{body(phase_b)}

## 5. Source–Label Confounding

{body(phase_c)}

![Source label composition](figures/source_label_composition.png)

## 6. Sensor & Acquisition Heterogeneity

{body(phase_d)}

## 7. Preprocessing Fidelity

{md(fidelity[['float32_cache_exact','index_cache_exact','packed_float16_exact','independent_masks_exact','sample_coverage_exact']].agg(['count','sum']).reset_index().rename(columns={'index':'check'}))}

Cache comparison là exhaustive cho recording cohort; summary statistical tests dùng participant units. Normalization replay dựa trên statistics float32 gốc, trong khi input inference được half-packed rồi lên float32. Clip/quantization max không tự đo clinical information loss. Không chứng minh toàn bộ Kinematic feature matrix từng bit bằng cách trích xuất lại1036features cho cả cohort; baseline X/hash/source code được bảo vệ và các đường input được đối chiếu, đây là phạm vi audit được công bố.

## 8. Existing Model Performance Under Source Confounding

{body(phase_e)}

![Fusion corrected/harmed by source](figures/fusion_effects_by_source.png)

## 9. Shortcut Learning & Leakage Assessment

**Verified code facts:** feature extraction nhận signals, không cognitive scores hoặc participant ID làm vector đầu vào. IDs dùng tra cứu file, join và partition; task identity embeddings là task identity, không participant identity. Baseline pipelines fit imputer/variance/scaler trên Train. Temporal fit/normalizer IDs khớp Train theo saved provenance; checkpoint hashes và protected artifacts được kiểm tra. Source-only diagnostic được fit trên Train labels đúng phạm vi.

**Statistical association:** city/site/date và acquisition summaries liên quan nhãn; metadata source-only dự đoán group mạnh. **Hypotheses:** mô hình có thể khai thác sensor/population/protocol shortcuts hoặc motor signal, hoặc cả hai. **Unidentifiable:** tỷ phần performance do mỗi nguồn, hardware causation, diagnosis truth của disagreement. Không có randomized/crossover/source-balanced design để phân rã.

**Researcher-level selection bias:** cohort đã được dùng trong nhiều feature/embedding/architecture decisions và partial OOF đã được xem trước khi khóa trọng tâm nghiên cứu. Train/Test participant isolation không làm quá trình chọn hướng độc lập với cohort. Bootstrap conditional không sửa selection bias này.

## 10. Statistical Identifiability & Validation Feasibility

{body(phase_f)}

## 11. Required Additional Data/Metadata

Codebook và assessment provenance (clinical criteria, assessor, instrument/threshold/date, co-occurring conditions), recruitment/source definitions, omission derivation and official retained IDs, duplicate clinical rows, raw/processed ID mapping, device model/serial/firmware/driver/software, timestamp units and resolution, pressure/angle units and calibration, SOP/operator/task instructions, acquisition order/dates, repeat-subject linkage and privacy-safe source identifiers. Cần cases và controls tại cùng nhiều nguồn, không chỉ thêm nhiều người từ các nguồn đơn lớp hiện có.

## 12. Implications for Stroke Temporal Transformer

Không rebuild network. Bounded segment attention không phải clinical-stroke annotation hoặc full-time-series attention; learned task pooling không phải task Transformer/duration pooling. Các derivative/time/acquisition channels cũng có thể chứa source signatures. Relative attention weights không phải causal explanation. C1 hiện là hypothesis có development controls chưa đủ capacity/budget-matched evidence; C2 chưa được xác nhận như motor complementarity vì fixed fusion không tăng toàn bộ F1 và nguồn/clinical ascertainment còn confounded. C3 cần dữ liệu độc lập.

## 13. Research GO / NO-GO Decisions

{md(decisions)}

Không có experiment mới để tối ưu F1 được khởi chạy. Metadata negative control chỉ là diagnostic đã yêu cầu, không thay clinical model.

## 14. Limitations

Không có clinical codebook/SOP hoặc chứng chỉ units; RDS metadata chưa giải mã vì thiếu local reader, có thể chứa attributes/labels không xuất sangCSV. Không truy cập thêm dữ liệu mạng. Statistical tests không model school/source sampling dependence; city1 ít positives; class/source imbalance; sparse contingency cells; no external test. Không xác minh physical-person identity từ ID alone. OOF replay không phải fresh neural inference. Không task-level attribution từ participant OOF; quality-by-task chỉ là mô tả. Không xác định trường nhãn nào đúng lâm sàng. Các blocker/unverified items được giữ trong `BLOCKERS.md`, không điền số liệu giả.

## 15. Reproducibility & Artifact Manifest

Reference commit `{context['reference_commit']}`, actual start `{context['actual_commit']}`. {context['branch_not_created_reason']}. CLI/seed/phạm vi trong `REPRODUCE.md`; snapshot SHA trước audit trong `PROTECTED_BEFORE.csv`; bảo toàn cuối và unit tests trong `VALIDATION_CHECKS.json`; hashes mọi artifact mới trong `ARTIFACT_MANIFEST.json`. Kiểm toán hoàn thành về mặt tính toán không có nghĩa clinical validity được xác nhận. Không push/merge.

Một số checkpoint ignored của XAI ngoài phạm vi directory scan ban đầu còn có expected hashes trong historical protected manifests. `LEGACY_PROTECTED_ADDITIONAL.csv` đối chiếu trực tiếp với các hashes lịch sử đó; không sửa hoặc bổ sung hồi tố snapshot ban đầu. Validation kết hợp cả snapshot trước audit và các historical expectations, phân biệt nguồn bằng chứng rõ ràng.

Sự cố vận hành được ghi minh bạch trong `INCIDENT_AND_RESTORATION.json`: một lần gọi CLI report đã resolve nhầm module `scripts/report.py`, ghi lại9 tệp báo cáo/notebook/lock cũ. Không fit model hoặc ghi dữ liệu nguồn. Chín tệp được khôi phục byte-for-byte theo SHA snapshot trước audit (kể cả line endings), CLI đã dùng explicit local-module path guard và regression test. Kiểm tra bảo toàn cuối phải khớp toàn bộ snapshot, không chỉ9 tệp đã khôi phục. Không che giấu sự cố bằng cách cập nhật snapshot chuẩn.
'''
    text('COMPREHENSIVE_DATA_AUDIT_VI.md',comprehensive)
    text('BLOCKERS.md','''# Blockers và giới hạn không thể tự giải quyết

- Không thiếu SVC/cache/OOF cần thiết cho cohort hiện tại nếu D_STATUS và EF_STATUS complete.
- Thiếu local clinical codebook, recruitment SOP, official cohort/omit derivation và thiết bị/units/calibration metadata. Không thể xác minh clinical semantics hoặc nguyên nhân confounding; cần chủ dữ liệu.
- Hai RDS hiện diện: `data/raw_data/cognitive_test_data/cognitiveTestRdata.rds` và `data/processed_data/cognitive_test_data/cognitiveTestRdata_cleaned.rds`. Không có local pyreadr/Rscript; chỉ inventory/hash, chưa đọc attributes/value labels. CSV/merged vẫn đủ cho các phép đối chiếu nhị phân và counts đã công bố. Không cài/tải qua mạng.
- Source-link anomalies ngoài cohort và duplicate clinical ID cần giải thích, không tự nối/sửa.
- Participant OOF không chứa task-specific attributions; không xác định task gây correction. Không chạy inference ablation mới.
- Working tree có4 untracked files cũ; điều kiện clean-tree để tạo nhánh audit chưa thỏa. Giữ nguyên nhánh, không stash/xóa/commit/push.
''')
    text('REPRODUCE.md','''# Reproduce — local read-only audit

Environment: project `.venv`, Python3.10+, NumPy/pandas/SciPy/scikit-learn/matplotlib/pytest đã cài. Không network, GPU hoặc checkpoint deserialization. Không cần cài package mới.

```powershell
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase all
```

Các pha độc lập sau initialization:

```powershell
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase abc
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase d
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase ef
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase report
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase tests
.venv/Scripts/python.exe scripts/data_audit_v3/run.py --phase validate
```

Seed42; metadata smoothing2 cố định, permutation999, paired bootstrap2000, calibration10 equal-width bins. No hyperparameter/threshold search. Initial protected snapshot không ghi đè khi rerun; audit outputs riêng có thể tái tạo, original artifacts không sửa. Source metadata numeric/blank semantics được ghi; literal prefix NA được giữ.

PhaseD exhaustive4000raw files và3612cohort cached recordings; có thể tốn vài phút CPU/I/O. PROTECTED_BEFORE và final verification đọc hàng chụcGiB chỉ để hash, không copy dữ liệu. Figures PNG300dpi/SVG, participant units, all source units uncalibrated. Runtime actual trong phase status files. Manifest SHA của artifact có thể khác giữa reruns do timestamps/figure metadata; kiểm tra số liệu và source hashes, không yêu cầu byte-identical report timestamps.
''')
    print('VIETNAMESE AUDIT REPORTS AND FIGURES COMPLETE',flush=True)


if __name__=='__main__':main()
