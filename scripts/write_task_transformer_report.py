"""Generate the Vietnamese research report from completed experimental tables."""
import json
import numpy as np
import pandas as pd
from task_transformer_common import ROOT,OUT,FAMILIES

T=OUT/'tables'
def read(name):return pd.read_csv(T/(name+'.csv'))
def table(frame):
    def fmt(v):return f'{v:.4f}'if isinstance(v,(float,np.floating))else str(v).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v)for v in row)+' |'for row in frame.itertuples(index=False,name=None))
def main():
    primary=read('ablation_primary').set_index('model');base=primary.loc['lightgbm'];nested=primary.loc['nested_selected']
    names=['lightgbm','lightgbm_without_text',*FAMILIES,'nested_selected'];summary=[]
    for name in names:
        d=primary.loc[name];r={'Mô hình':name}
        for col in ('f1_macro','accuracy','auc','balanced_accuracy'):r[col+' OOF']=d[col];r[col+' fold']=f"{d['mean_'+col]:.4f} ± {d['std_'+col]:.4f}"
        summary.append(r)
    matched=read('matched_ablation');paired=read('paired_comparisons');rob=read('robustness_summary')
    st=read('attribution_stability').groupby(['model','level'])[['spearman','top3_jaccard']].mean().reset_index()
    faith=read('faithfulness');importance=read('attribution_importance');seed=read('seed_sensitivity_summary')
    configs=read('selected_configurations');cost=read('inference_cost').groupby('model').median(numeric_only=True).reset_index()
    cost['inference_ms_per_person']=1000*cost.median_inference_seconds/43
    costs=primary.reset_index()[['model','parameters_min','parameters_max','train_seconds_mean','task_count_mean']].merge(cost[['model','inference_ms_per_person','timing_only_train_seconds']],on='model',how='left')
    overfit=read('overfitting_diagnostics').groupby('family')[['best_epoch','loss_gap','train_accuracy','validation_accuracy']].mean().reset_index()
    top=importance[(importance.level=='task')&importance.model.isin(['lightgbm','transformer','motor_normalized','xai_selected'])].sort_values(['model','rank']).groupby('model').head(5)
    groups=importance[(importance.level=='group')&importance.model.isin(['lightgbm','transformer','motor_normalized'])].sort_values(['model','rank']).groupby('model').head(5)
    nestedchoices=configs[configs.family=='nested_selected'][['fold','source_model','candidate']]
    attention=read('attention_agreement');comparison=read('xai_baseline_rank_comparison');mask=read('masking_summary')
    cache=json.loads((OUT/'cache/validation.json').read_text());pilot=json.loads((OUT/'pilot.json').read_text())
    ig=[];refinements=[]
    for p in (OUT/'analysis').rglob('*_ig*.npz'):
        with np.load(p)as a:
            ig.append({'steps':int(a['steps']),'maximum_absolute_residual':float(abs(a['error']).max())})
            if 'refined_indices'in a:refinements.append(dict(artifact=p.relative_to(OUT).as_posix(),participants=len(a['refined_indices']),panel_order=int(a['panel_order'])))
    ig=pd.DataFrame(ig)
    rq1=matched.set_index('comparison').loc['self_attention']
    rq2=matched.set_index('comparison').loc['motor_normalization']
    rqaug=matched.set_index('comparison').loc['training_augmentation']
    reduction=primary.loc['xai_selected'];notext=primary.loc['without_text_tasks']
    text=f'''# Motor-Invariant Kinematic Task Transformer — báo cáo thực nghiệm

Trạng thái số liệu: hoàn tất huấn luyện và phân tích sáu outer folds; chỉ coi toàn bộ deliverables hoàn tất khi `validation.json` có `complete: true`. “Motor-invariant” là **giả thuyết nghiên cứu**, chưa phải thuộc tính đã chứng minh. Nghiên cứu chỉ dùng Kinematic, không triển khai Fusion hay cross-language.

## Kết quả chính

Baseline Kinematic + LightGBM được giữ nguyên: Accuracy OOF **{base.accuracy:.4%}**, macro-F1 **{base.f1_macro:.4f}**, ROC-AUC **{base.auc:.4f}**. Bộ chọn cấu hình neural hoàn toàn bằng inner CV (`nested_selected`) đạt Accuracy **{nested.accuracy:.4%}**, macro-F1 **{nested.f1_macro:.4f}**, AUC **{nested.auc:.4f}**; chênh lệch macro-F1 OOF so với baseline **{nested.f1_macro-base.f1_macro:+.4f}**. Đây là kết quả của quy trình chọn họ mô hình trong Train, không phải chọn họ có outer Test cao nhất sau khi xem bảng.

{'Chưa có bằng chứng để thay thế LightGBM bằng Transformer: nested neural thấp hơn baseline về macro-F1 trên bộ dữ liệu này.' if nested.f1_macro<=base.f1_macro else 'Nested neural có macro-F1 OOF cao hơn trong cohort này; chưa đủ để tuyên bố vượt trội tổng quát hoặc lâm sàng, cần kiểm chứng độc lập.'}

{table(pd.DataFrame(summary))}

OOF gộp 258 người khác với trung bình sáu fold; SD theo sáu fold dùng ddof=1, không phải khoảng tin cậy. Chi tiết toàn bộ đối chứng trong `tables/ablation_primary.csv`; ba seeds trong `tables/seed_sensitivity.csv`.

## Kiểm toán và dữ liệu

Làm việc trên branch `study/kinematic-task-transformer`, kế thừa project hiện có. Tái sử dụng nguyên extractor `diagramo.features.kinematic`, cache `data/derived/kinematic.npz`, mapping `schema.json`, fold assignments và checkpoint LightGBM cũ. 258 participants, nhãn `group` (115/143), 14 tác vụ × 74 đặc trưng = 1.036 chiều. Mapping thực tế ở `feature_mapping.csv`; không tự sắp xếp tác vụ theo tên hoặc giả định trình tự thời gian.

Đã tái trích xuất nguyên bản cho toàn bộ 258 người và kiểm tra **khớp chính xác** cache cũ. {cache['transformed_banks']} banks biến đổi không dùng nhãn, {cache['checked_raw_record_conditions']:,} kiểm tra raw-record/condition; các JSON nguồn có SHA256. Artifact lịch sử được bảo vệ bằng `protected_manifest.json` và xác minh lại lúc validation. Task 1 và pipeline bốn nhánh không bị sửa.

## Kiến trúc và giao thức tránh leakage

Mỗi task là một token 74 chiều. Median imputation giữ đủ 74 tọa độ và StandardScaler chỉ fit trên Train tương ứng. Shared Linear 74→32/64 + GELU + dropout; optional task-ID embedding; 1–2 pre-LayerNorm self-attention blocks, 2/4 heads, FFN chiều 2d; mean hoặc learned attention pooling; LayerNorm và binary logit. Không có chronological positional encoding. Task-MLP dùng shared token feed-forward blocks và mean pooling, không có attention. Missing tokens bị triệt tiêu ở input, attention keys và pooling. Các mạng được train với task dropout 0,1, bảo đảm còn ít nhất một task.

Quy trình cố định outer Stratified 6-Fold seed 42, mỗi Test 43 người, Train 215. Inner Stratified 3-Fold chỉ trong outer Train. Mỗi family có bốn ứng viên cố định (`configs/task_transformer.json`): d32/no task-ID/mean; d32/task-ID/mean; d64/task-ID/mean; d32/hai layers/task-ID/attention pool. Task-MLP luôn mean pooling. Tối đa 50 epochs, patience 7, chạy ít nhất 10; AdamW lr 0,001, weight decay 0,001, batch64, gradient clipping1. Chọn checkpoint theo inner validation BCE; chọn cấu hình theo mean inner macro-F1, tie-break AUC, ít tham số, thứ tự cấu hình. Final model dùng toàn bộ outer Train và số epochs là median best epoch của ba inner runs; **không early-stop trên outer Test**.

432 inner neural fits; 144 final neural fits gồm 108 fits của sáu families × ba seeds × sáu folds và 36 matched controls seed42. 36 tree model records gồm sáu checkpoint baseline tái sử dụng và 30 fits mới (hai seed bổ sung/full tasks, ba seed/without text). XAI task selection tái cross-fit ba lần trong từng inner Train: 54 LGBM fits bổ sung; final outer Train chỉ tái sử dụng Task-1 ranking khi IDs và hash đầu vào khớp. Sáu LGBM fits khác chỉ đo thời gian và phải khớp dự đoán baseline, không dùng chọn mô hình. Pilot 10 epochs chỉ trên outer Train0 để kiểm tra tài nguyên, không dùng Test chọn grid.

Rank top-k=7 hoặc10 chỉ từ Train; trong grid, top-k còn đi cùng thay đổi kiến trúc nên không suy diễn tác động riêng top-k từ grid. Có matched top10/full14 cùng cấu hình để hỗ trợ kiểm chứng. Ba seeds 42/43/44 giữ cấu hình được chọn bằng seed42 và fold assignments; đây là sensitivity của huấn luyện, không phải 18 folds độc lập hoặc ba cohorts.

Các lựa chọn cuối cùng của nested neural:

{table(nestedchoices)}

**Giới hạn cấp nghiên cứu:** Task-1 XAI trên chính cohort và outer folds này đã được xem để hình thành giả thuyết. Tính train-only của code ngăn leakage trực tiếp, nhưng không xóa adaptive bias do người nghiên cứu đã xem kết quả trước đó. Kết quả này là exploratory reuse; cần external holdout để kiểm chứng khẳng định mới.

## Nuisance và thông tin có thể mất

Motor normalization là per-record không dùng nhãn: trừ điểm đầu ở x/y, chia chung theo RMS bán kính quỹ đạo quanh điểm đầu; pressure chia percentile95 của trị tuyệt đối. Guard tránh chia cho0. Giữ nguyên thời gian thực, pen state, tilt, azimuth. Nó hướng tới thay đổi vị trí, kích thước và gain pressure, nhưng cũng có thể mất kích thước chữ, biên độ vận động và mức lực liên quan impairment. Không gọi mọi biến thiên này là nhiễu vô hại.

| Biến đổi raw | Train augmentation / held-out stress | Thông tin có thể mất hoặc làm thay đổi |
|---|---|---|
| Translation | ±5% / ±10% bounding-box span ở hai trục | Vị trí/offset tuyệt đối; không đồng nghĩa đổi nội dung |
| Spatial scale | ×1,05 / ×1,10 quanh điểm đầu | Kích thước, vận tốc/acceleration tuyệt đối |
| Pressure gain | ×1,05 / ×1,10 | Mức lực có thể mang tín hiệu; chưa mô phỏng clipping/ADC |
| Sampling half | Giữ cách một sample và điểm cuối | Mất cao tần; không anti-alias; không tương đương mọi thiết bị |
| Timing warp | sin với biên độ1% /2% duration | Nhịp cục bộ thay đổi, monotonic derivative vẫn dương |
| Duration scale | ×1,05 / ×1,10 time từ thời điểm đầu | Tốc độ, thời lượng vốn có thể quan trọng cho nhãn |

Mỗi biến đổi gọi lại extractor gốc cho cả 14 tasks, kiểm tra finite và tính hợp lệ thời gian. Các stress được cố định trước phân tích và không dùng chọn cấu hình. Riêng `sampling_half` có cùng cường độ Train/Test; đây không phải thử ngoại suy toàn bộ nuisance. Train-augmented chọn bank ngẫu nhiên cho từng participant với xác suất0,3, chỉ index current Train; validation/test không augment. Các biến đổi là sensitivity hypotheses, **chưa chứng minh label-preserving**. Cache toàn cohort chỉ thực hiện biến đổi xác định không nhãn; mọi imputer/scaler/ranker/model vẫn fit Train riêng.

## Ablation kiểm soát và so sánh paired

{table(matched)}

Các matched controls không tham gia chọn nested winner. Cùng d/head/layer/pooling khi so sánh attention/normalization/augmentation/selection; epoch vẫn do inner validation của từng phương án. Task-MLP và Transformer có số tham số khác nhau, vì vậy so sánh attention chưa phải can thiệp nhân quả chỉ đổi duy nhất một cơ chế/capacity.

{table(paired[paired.model.isin(names)][['model','corrected','harmed','unchanged_correct','unchanged_wrong','mean_delta_f1_macro','std_delta_f1_macro']])}

Corrected = baseline sai, mô hình mới đúng; harmed = baseline đúng, mô hình mới sai. Cùng 258 participants, threshold probability >0,5 giữ nguyên, không chọn ngưỡng lại từ Test. Chênh lệch theo fold là paired; không dùng các fold phụ thuộc như những cohort độc lập để khẳng định significance.

## Overfitting, seeds và chi phí

{table(overfit)}

Loss gap là validation BCE trừ Train BCE tại inner best epoch; Train loss được tính trong chế độ train có dropout/task masking nên gap không phải đo độc lập hoàn hảo về overfitting. Learning curves là mô tả, các epoch muộn có ít runs do early stopping. Outer model chỉ có Train curve; không có outer-Test loss curve dùng điều khiển fit.

{table(seed)}

{table(costs[costs.model.isin(names)])}

Neural parameter count và tree complexity không đồng nhất: không gán cây một số neural parameters giả. Train cost neural là mỗi final fit và inference là median20 warm measurements cho batch43, gồm cached features→preprocessor→GPU transfer→probability. LightGBM chạy CPU, neural trên Quadro T1000; không diễn giải đây là so sánh tốc độ công bằng phần cứng. Raw preparation cho toàn bộ 19 banks mất {cache['seconds']:.1f}s với ba CPU workers, không nằm trong inference latency. Pilot kiểm tra fit GPU với batch64; chi phí pipeline thực gồm inner search, ranker, cache và attribution, không chỉ một forward pass.

## Robustness khi nuisance và thiếu tác vụ

{table(rob[rob.model.isin(['lightgbm','transformer','motor_normalized','train_augmented'])][['model','condition','delta_f1_macro_mean','delta_f1_macro_std','mean_absolute_probability_shift_mean','attribution_rank_spearman_mean']])}

Delta là stress trừ clean trên cùng held-out fold. Probability shift nhỏ nhưng clean performance thấp không chứng minh mô hình tốt hơn. Attribution rank stress chỉ tính cho LightGBM, raw Transformer và motor-normalized Transformer; ô trống ở family khác không phải correlation0. Kiểm tra riêng spatial invariance của phép chuẩn hóa ở raw arrays trong unit tests không thay thế kết quả dự đoán thực.

{table(mask[['model','kind','drop_f1_macro_mean','drop_f1_macro_std']])}

Đã thử14 single-task masks, copy+dictation mask, và 2/4/6 tasks với10 subset draws/fold; dùng cùng subset cho mọi mô hình. Nets dùng missing-token mask, trees thay bằng median Train: khác cơ chế nên so sánh có giới hạn. Tasks đã không được model chọn có thể bị mask mà không đổi prediction; không được suy thành độ bền với cùng số task đang hoạt động bị mất. Kiểm tra độc lập zero-feature ablation giữ token present và đặt Z=0 (Train mean sau scaling); trees dùng Train raw mean. Hai phép kiểm tra này được lưu riêng.

## XAI: logit IG, task/group và kiểm chứng độc lập

IG dùng positive-class **logit**, reference là vector chuẩn hóa0 tương ứng mean Train; giữ cùng task mask và task-ID embeddings. Tổng attribution khớp `logit(x)-logit(reference)` với tolerance0,002 +0,002×|delta logit| trên mỗi participant. Dùng Gauss–Legendre quadrature32→512 bước, tăng cho tới khi hội tụ, không đổi ngưỡng theo Test. {len(ig)} artifacts clean/stress, tối đa residual tuyệt đối {ig.maximum_absolute_residual.max():.6f}; số bước thực tế từ {ig.steps.min()} đến {ig.steps.max()}. Attribution không phải xác suất và không tự quy đổi từng chiều qua sigmoid. TreeSHAP tái dùng exact saved baseline và kiểm tra raw-score additivity; references giữa TreeSHAP và IG khác nhau, so sánh rank/share, không coi absolute mass là cùng chuẩn đo sinh học.

Mapping task và12 feature groups lấy đúng Task-1 API, bao gồm velocity, acceleration, jerk, pressure, orientation, geometry/curvature, timing, pen-state và FFT. Một số khái niệm như smoothness được phản ánh gián tiếp qua jerk/geometry; không gán nhãn sinh học mới cho feature chưa có phép đo tương ứng. CSV group có `n_features` và mass/feature để thấy ảnh hưởng kích thước nhóm.

Top5 tasks theo từng mô hình:

{table(top[['model','block','share_percent','rank','fold_sd']])}

Top5 feature groups:

{table(groups[['model','block','n_features','share_percent','mass_per_feature']])}

Độ ổn định giữa15 cặp outer folds (mean Spearman/Top3 Jaccard; các cặp phụ thuộc):

{table(st)}

Faithfulness: rank attribution so với macro-F1 drop khi grouped permutation (10 repeats/task/group/fold) và zero-feature ablation. Có12.480 held-out permutation evaluations, dùng chung hoán vị giữa mô hình. Negative importance/drop được giữ nguyên. Perturbation có thể ra ngoài distribution hoặc chịu feature redundancy; correlation thấp phải được báo cáo, không sửa ranking để phù hợp kỳ vọng.

{table(faith)}

Rank disagreement với LightGBM:

{table(comparison)}

Attention chỉ là chẩn đoán phụ, không tự coi là explanation. Incoming attention được trung bình qua layers/heads và valid queries; pooling weights lưu riêng, missing keys0. Attention-vs-IG:

{table(attention)}

Các ca local trong `tables/local_cases.csv` chọn ID nhỏ nhất theo thứ tự từ điển cho từng outcome của raw Transformer, sau khi hoàn tất dự đoán. Đây là minh họa hậu nghiệm, không dùng để đổi model. Có bảng signed/absolute task IG cho toàn bộ participants để kiểm tra cả đúng, sai, corrected và harmed; không chỉ trình bày ca đẹp.

## Trả lời câu hỏi nghiên cứu

**RQ1 — Self-attention có cải thiện tương tác task?** Matched attention so với Task-MLP: Δmacro-F1 OOF {rq1.oof_delta_f1_macro:+.4f}; paired fold mean {rq1.mean_delta_f1_macro:+.4f} ± {rq1.std_delta_f1_macro:.4f}. Đây là bằng chứng giới hạn trong grid và cohort này; {'không thấy lợi ích dự đoán từ matched attention.'if rq1.oof_delta_f1_macro<=0 else 'có lợi ích mô tả so với matched MLP, nhưng chưa chứng minh vượt baseline tree hoặc quan hệ nhân quả giữa tasks.'}

**RQ2 — Motor processing có bền mà không mất thông tin?** Matched normalization Δmacro-F1 OOF {rq2.oof_delta_f1_macro:+.4f}, augmentation {rqaug.oof_delta_f1_macro:+.4f}. Cần đọc đồng thời clean metrics và nuisance table: phép normalize có thể triệt scale/gain theo thiết kế nhưng lấy đi thông tin dự đoán. Không gọi nó là cải thiện miễn phí; không thay baseline chỉ từ invariance toán học.

**RQ3 — Giải thích ổn định/faithful hơn?** Xem riêng hai bảng stability và faithfulness ở trên; không gộp thành một tiêu chí thuận lợi duy nhất. IG completeness chỉ chứng minh phép tích phân khớp đầu ra, không chứng minh clinical validity. Stability cao hơn nếu có cũng không bù được dự đoán thấp hơn; attention khác IG là kết quả cần phân tích, không phải lý do ép rank khớp.

**RQ4 — Ít tasks có cạnh tranh?** XAI-selected dùng trung bình {reduction.task_count_mean:.2f} tasks, macro-F1 {reduction.f1_macro:.4f} (Δbaseline {reduction.f1_macro-base.f1_macro:+.4f}); without copy/dictation dùng12 tasks đạt {notext.f1_macro:.4f} (Δbaseline {notext.f1_macro-base.f1_macro:+.4f}). Kết quả tree without-text được báo cáo song song. Không kết luận số tasks tối thiểu từ SHAP/IG hoặc một vài k; cần task-ablation protocol riêng trên dữ liệu độc lập. Bỏ copy/dictation chỉ kiểm tra phụ thuộc hai task nhiều nội dung chữ, không phải kiểm định language invariance.

**RQ5 — Generalization dưới nuisance?** Chỉ chứng minh mức nhạy thực nghiệm với sáu biến đổi cố định đã nêu, không phải generalization tới thiết bị, bệnh viện, ngôn ngữ hoặc cohort mới. Timing/sampling có thể tác động ngược nhãn; phương án robust nhưng accuracy thấp không tự là mô hình phù hợp hơn. Không chọn lại nuisance magnitudes/thresholds/model từ các outer stress results.

## Hình và bảng nên giữ cho paper

Giữ architecture (01), predictive ablation (02), matched ablation (03), seeds (04), curves (05), compute (06), nuisance (07), missing tasks (08), task/group attribution (09), stability (10), faithfulness (11), confusion (13), corrected/harmed (14). Hình12 attention đặt supplementary và ghi rõ không phải faithful attribution. Mọi hình có PNG300dpi/SVG, caption/nguồn/hashes trong `figures/manifest.json`. Đây là paper drafts; phần diễn giải phải giữ negative results và giới hạn exploratory cohort reuse.

Notebook `notebooks/06_kinematic_task_transformer.ipynb` ưu tiên các **cell bảng số kết quả cuối cùng**, sau đó mới hiển thị hình. Các CSV chính: `ablation_primary`, `matched_ablation`, `fold_results`, `oof_predictions`, `paired_comparisons`, `robustness_summary`, `masking_summary`, `attribution_importance`, `attribution_stability`, `faithfulness`, `seed_sensitivity_summary`, `inference_cost`. Không có chỉ số do ảnh tạo sinh hoặc ước lượng từ biểu đồ.

## Tái lập và validation

Chạy bằng môi trường hiện có, từ project root:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/prepare_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/train_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/train_task_transformer_controls.py
.venv/Scripts/python.exe -X utf8 -B scripts/analyze_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/report_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/write_task_transformer_report.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_task_transformer_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/validate_task_transformer.py --numerics-only
.venv/Scripts/python.exe -X utf8 -B scripts/plot_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 06_kinematic_task_transformer.ipynb --validation-output results/task_transformer/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_task_transformer.py
```

Raw data/cache/checkpoints không đưa lên Git; checkpoint Task2 ở `results/task_transformer/checkpoints/`, được giữ local, hash được lưu trong JSON. Các bước fit resumable theo training signature; không sửa config/code huấn luyện rồi âm thầm dùng checkpoint cũ. `numeric_validation.json` tái chạy inference từ mọi checkpoint, kiểm tra Train medians/scaler, IDs, choices/epochs inner, baseline OOF, nuisance/masks/zero-feature/permutation probabilities và IG logits/completeness; đóng băng hash artifacts và code. Final validation còn kiểm tra tám unit tests, notebook execution, hình/nguồn và lịch sử không đổi.

Đề xuất thực tế: giữ baseline LightGBM làm mốc chính; giữ pipeline Transformer nhẹ và các kết quả âm/dương như nghiên cứu ablation. Chỉ xem xét thay đổi hướng mô hình tiếp theo sau review, không triển khai thêm kiến trúc, Fusion, Knowledge Graph hay cross-language trong nhiệm vụ này. Giải thích mô hình không phải kết luận nhân quả hoặc chẩn đoán lâm sàng.
'''
    refinement_note='''
### Kiểm tra số học đặc trưng cực đoan

Trong outer fold4, participant NB05301 có một tọa độ sau Train standardization khoảng 2,64×10^13. Giữ nguyên extractor, scaler, checkpoint và prediction; không clip lại sau khi xem Test. Global Gauss–Legendre512 bỏ lỡ chuyển tiếp logit rất sát alpha0, tạo residual IG khoảng1,629 ở Task-MLP. Đã bổ sung quadrature trên geometric alpha panels cho đúng các participant chưa hội tụ; không nới tolerance và không dùng nhãn để chọn phương pháp tích phân. Có unit test tanh với input10^13 kiểm tra độc lập. Chênh lệch input cực đoan này cũng cho thấy scaling và neural generalization có giới hạn; nên nghiên cứu preprocessing robust bằng protocol mới trong tương lai, không sửa mô hình của thực nghiệm đã khóa.

Các artifact phải tinh chỉnh tích phân (CSV/NPZ còn lưu global residual ban đầu, indices, panel order/count):

'''+table(pd.DataFrame(refinements))+'\n'
    taskst=st[st.level=='task'].set_index('model')
    taskfaith=faith[(faith.level=='task')&(faith.comparison=='attribution_vs_permutation')].set_index('model')
    interpretation=f'''### Diễn giải định lượng bổ sung

Raw Transformer có mean task-rank stability rho={taskst.loc['transformer','spearman']:.4f}, thấp hơn LightGBM {taskst.loc['lightgbm','spearman']:.4f}; agreement attribution–permutation cũng thấp hơn ({taskfaith.loc['transformer','spearman']:.4f} so với {taskfaith.loc['lightgbm','spearman']:.4f}). Chưa có bằng chứng Transformer giải thích ổn định hoặc faithful hơn baseline. XAI-selected có Top3 Jaccard cao hơn ({taskst.loc['xai_selected','top3_jaccard']:.4f}), nhưng zero-attribution ở tasks bị loại và lựa chọn task từ LightGBM có thể làm agreement/stability cao theo thiết kế; không coi đó là cải thiện giải thích độc lập hoặc lý do bỏ qua accuracy thấp hơn.

Motor-normalized giữ macro-F1 không đổi trước translation/spatial-scale/pressure-gain đã thử, trong khi raw Transformer vẫn nhạy với một số biến đổi; probability shifts không hoàn toàn0 do floating point và extractor. Đổi lại clean macro-F1 của motor-normalized ({primary.loc['motor_normalized','f1_macro']:.4f}) thấp hơn raw Transformer ({primary.loc['transformer','f1_macro']:.4f}) và baseline ({base.f1_macro:.4f}); matched normalization cũng giảm điểm. Sampling half là stress mạnh: baseline giảm mean fold F1 khoảng0,2490, raw Transformer0,2527, motor-normalized0,0806, train-augmented0,0922. Đây là độ giảm tương đối; đọc cả mức điểm tuyệt đối, không kết luận tổng quát rằng neural tốt hơn.

XAI-selected có OOF F1 cao hơn raw Transformer trong bảng family, nhưng matched top10 so với full14 cùng cấu hình giảm F1; chưa thể gán lợi ích family cho riêng task selection. Task-ID embedding cũng giảm F1 trong matched comparison. Matched augmentation có tín hiệu tích cực nhưng không được chọn làm “winner đã định trước” bằng cách nhìn outer Test; giữ nó là giả thuyết cho kiểm chứng độc lập.

'''
    text=text.replace('## Hình và bảng nên giữ cho paper',refinement_note+'\n'+interpretation+'## Hình và bảng nên giữ cho paper').replace('tám unit tests','chín unit tests')
    text=text.replace('Train loss được tính trong chế độ train có dropout/task masking nên gap không phải đo độc lập hoàn hảo về overfitting.', 'Train loss/accuracy được đánh giá lại trong chế độ eval trên clean Train, không có dropout/task masking; loss gap vẫn chỉ là chẩn đoán trên tập Train/inner-validation nhỏ.')
    text=text.replace('Train cost neural là mỗi final fit', 'Train cost neural đo vòng tối ưu của mỗi final fit, không gồm fit preprocessor/khởi tạo')
    text=text.replace('Sáu LGBM fits khác chỉ đo thời gian', 'Sáu LGBM fits khác chỉ đo thời gian')
    (OUT/'REPORT_VI.md').write_text(text,encoding='utf-8');print('VI REPORT GENERATED')

if __name__=='__main__':main()
