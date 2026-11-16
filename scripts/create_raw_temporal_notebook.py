"""Read-only numerical-results-first notebook for the replacement Task 2."""
import nbformat as nbf
from raw_temporal_common import ROOT
def main():
    cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x))
    def code(x):cells.append(nbf.v4.new_code_cell(x))
    md('# Raw Temporal Graphomotor Learning — Task 2 thay thế\n\nRaw chronological samples → Local movement → Stroke/segment → Task → Participant. Notebook chỉ đọc artifacts; không train hoặc tối ưu khi mở. Bảng số đặt trước hình. [Báo cáo tiếng Việt](../results/raw_temporal/REPORT_VI.md).')
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()
if not (ROOT/'results/raw_temporal').exists(): ROOT=ROOT.parent
OUT=ROOT/'results/raw_temporal';T=OUT/'tables'
pd.set_option('display.max_columns',40)
pd.set_option('display.max_rows',150)
pd.set_option('display.precision',6)
def read(name):return pd.read_csv(T/(name+'.csv'))
for name in ('data_validation','fit_validation','xai_validation'):
    v=json.loads((OUT/(name+'.json')).read_text());assert v['complete'];display(pd.DataFrame([v]))""")
    md('## Kết quả cuối cùng — OOF và fold mean±SD\n\nR1=CNN; R2=TCN; R3=flat Temporal Transformer; R4=Hierarchical. `nested_selected` chọn model trong inner CV; baseline LightGBM là protected reference. Independent n=258, không phải số windows.')
    code("d=read('primary_ablation')\ndisplay(d[['model','f1_macro','accuracy','auc','balanced_accuracy','sensitivity','specificity']])\ndisplay(d[['model','mean_f1_macro','std_f1_macro','mean_accuracy','std_accuracy','mean_auc','std_auc','mean_balanced_accuracy','std_balanced_accuracy','mean_sensitivity','std_sensitivity','mean_specificity','std_specificity']])")
    md('## Paired prediction comparison và confusion matrices\n\nCùng participants, threshold>0,5; không chọn lại bằng outer Test. Khác error cases chưa chứng minh fusion gain.')
    code("display(read('paired_comparisons'))\ndisplay(read('confusion_matrices'))")
    md('## Kiểm tra dự đoán một lớp\n\nNgưỡng cố định0,5. ROC-AUC phản ánh thứ hạng xác suất, có thể khác0,5 ngay cả khi mọi quyết định cùng một lớp. Bảng này chỉ chẩn đoán sau đánh giá, không dùng chọn lại threshold.')
    code("display(read('decision_diagnostics').query('seed==42'))")
    md('## Temporal-information và multi-scale ablation\n\n`no_explicit_time` vẫn giữ derived velocity theo Δt; `no_position` vẫn có CNN order và clock-valued sample inputs. `index_time` giữ integrity boundaries. Đọc đủ giới hạn trong báo cáo.')
    code("display(read('temporal_multiscale_ablation'))\ndisplay(read('temporal_perturbation_summary'))")
    md('## Từng fold, inner choices và seeds\n\nMean±SD mô tả6 folds, không là confidence intervals. Repeated seeds dùng cùng cohort và configurations, không là independent confirmation.')
    code("display(read('fold_results').query('seed==42').sort_values(['model','fold']))\ndisplay(read('selected_configurations'))\ndisplay(read('seed_sensitivity'))")
    md('## Raw data audit và full coverage\n\nKhông sort/drop timestamp anomalies, không resample whole tasks, không dùng engineered74 features. Đơn vị thời gian theo source convention, chưa có calibration certificate.')
    code("display(pd.DataFrame([json.loads((OUT/'audit/summary.json').read_text())]))\ndisplay(pd.read_csv(OUT/'audit/recordings.csv').groupby('task').agg(n=('n_samples','size'),samples_median=('n_samples','median'),samples_max=('n_samples','max'),duplicates=('duplicate_dt','sum'),large_gaps=('large_dt_over_1s','sum')).reset_index())\ndisplay(pd.read_csv(OUT/'audit/window_tradeoff.csv').groupby('patch_size')[['patches','raw_samples','padded_cells']].sum().reset_index())")
    md('## Overfitting và computational cost\n\nOnline Train curve có dropout/weights đang thay đổi; validation chạy eval. Tensor peak không gồm driver/context. Training chạy tuần tự một GPU worker; bảng latency đo warm CPU forward với hai threads, khác thiết bị và scope so với GPU inference cho toàn fold.')
    code("display(read('overfitting').groupby('model')[['best_epoch','loss_gap','max_epochs_reached']].mean().reset_index())\ndisplay(read('primary_ablation')[['model','parameters','parameters_max','train_seconds_mean','inference_seconds_mean','peak_gpu_bytes']])\ndisplay(read('latency').groupby('model').seconds.agg(['median','mean','std']).reset_index())")
    code("display(read('computational_cost_by_scope'))")
    md('## Task importance cho toàn bộ held-out participants\n\nOcclusion reliance là thay đổi output, không phải Shapley value hoặc clinical relevance. Kinematic SHAP chỉ đối chiếu mô tả, không dùng làm neural weights.')
    code("display(read('task_importance').query('seed==42').sort_values(['model','rank']))\ndisplay(read('kinematic_xai_rank_comparison'))\ndisplay(read('task_faithfulness'))")
    md('## Stroke và temporal intervals\n\nMapping về original pen-state run, down-stroke index, raw indices và timestamps. Bao gồm cả pen-up segments; không ghép thành trajectory liên tục giả.')
    code("display(read('behavioral_examples'))")
    code("s=read('stroke_occlusion');display(s.assign(absolute_effect=s.logit_delta.abs()).sort_values('absolute_effect',ascending=False).head(30))\ns=read('temporal_occlusion');display(s.assign(absolute_effect=s.logit_delta.abs()).sort_values('absolute_effect',ascending=False).head(30))")
    md('## Stability của explanations và representation geometry\n\nPairwise distances903/fold là các đại lượng phụ thuộc, không phải903 independent subjects. Không so sánh trực tiếp tọa độ latent giữa seeds/folds.')
    code("display(read('fold_xai_stability').groupby('model')[['spearman','top3_jaccard']].agg(['mean','std']).reset_index())\ndisplay(read('seed_xai_stability').groupby('model').spearman.agg(['mean','std']).reset_index())\ndisplay(read('representation_geometry_stability'))")
    md('## IG completeness, baselines, faithfulness và model randomization\n\nIG giải thích positive-class logit và dùng hai mathematical references. Completeness không chứng minh biological validity. Attention chỉ supplementary.')
    code("display(read('ig_validation'))\ndisplay(read('faithfulness'))\ndisplay(read('ig_baseline_seed_randomization'))\ndisplay(read('deletion_curves'))")
    md('## Ca TP/TN/FP/FN theo quy tắc cố định\n\nID nhỏ nhất theo từ điển cho mỗi outcome của hierarchical seed42 OOF. Không tạo ca cho category vắng mặt, không đổi threshold hoặc chọn ca đẹp.')
    code("display(read('selected_cases'))\ndisplay(read('oof_predictions').groupby(['model','seed']).agg(n=('participant_id','size'),unique_ids=('participant_id','nunique')).reset_index())")
    md('## Hình từ actual recordings và artifact tables\n\nPNG300dpi/SVG paper drafts. Không suy tremor, dysgraphia hoặc nhân quả từ attribution spikes.')
    code("manifest=json.loads((OUT/'figures/manifest.json').read_text())\nfor row in manifest:\n    display(Markdown('**'+row['name']+'** — '+row['caption']))\n    display(Image(filename=str(OUT/'figures'/(row['name']+'.png')),width=1000))")
    md('## Giới hạn và trạng thái bàn giao\n\nCohort/folds đã được xem ở các nghiên cứu trước: nested implementation không xóa adaptive research bias. Không external validation, cross-language hoặc fusion. Đánh giá architecture chưa capacity-match hoàn toàn, context bị giới hạn theo chunks. Đọc đầy đủ negative results, sự khác nhau giữa ablations và sensor assumptions trong báo cáo.')
    code("display(Markdown('Trạng thái hoàn tất sau notebook/tests: [validation.json](../results/raw_temporal/validation.json).'))\ndisplay(Markdown('Báo cáo: [REPORT_VI.md](../results/raw_temporal/REPORT_VI.md) — Audit: [audit/REPORT_VI.md](../results/raw_temporal/audit/REPORT_VI.md).'))")
    n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});nbf.validate(n);nbf.write(n,ROOT/'notebooks/06_raw_temporal_graphomotor.ipynb');print('RAW TEMPORAL NOTEBOOK CREATED')

if __name__=='__main__':main()
