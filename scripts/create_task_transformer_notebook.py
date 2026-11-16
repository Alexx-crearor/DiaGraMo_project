"""Build a read-only numerical-results-first notebook for Task 2."""
import nbformat as nbf
from task_transformer_common import ROOT

def main():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# Kinematic Task Transformer — kết quả và kiểm chứng\n\nNotebook chỉ đọc artifacts, không train lại. “Motor-invariant” là giả thuyết. Chỉ coi hoàn thành khi `results/task_transformer/validation.json` có `complete: true`. Bảng số được đặt trước hình. [Báo cáo tiếng Việt](../results/task_transformer/REPORT_VI.md).')
    code("""from pathlib import Path
import json, hashlib
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()
if not (ROOT/'results/task_transformer').exists(): ROOT=ROOT.parent
OUT=ROOT/'results/task_transformer'
TABLES=OUT/'tables'
pd.set_option('display.max_columns',40)
pd.set_option('display.max_rows',100)
pd.set_option('display.precision',6)
def read(name):return pd.read_csv(TABLES/(name+'.csv'))
validation=json.loads((OUT/'numeric_validation.json').read_text())
assert validation['complete']
for rel,digest in {**validation['artifact_sha256'],**validation['code_sha256']}.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
display(pd.DataFrame([validation['checks']]))""")
    md('## Bảng kết quả cuối cùng — seed42\n\nOOF tính trên258 người, mean±SD tính từ6 outer folds. SD không phải confidence interval. `nested_selected` được chọn hoàn toàn trong inner CV; các family khác là ablation.')
    code("d=read('ablation_primary')\ndisplay(d[['model','accuracy','f1_macro','auc','balanced_accuracy','mean_f1_macro','std_f1_macro','parameters_mean','task_count_mean']])\ndisplay(d[['model','mean_accuracy','std_accuracy','mean_f1_macro','std_f1_macro','mean_auc','std_auc','mean_balanced_accuracy','std_balanced_accuracy']])")
    md('## So sánh paired và các đối chứng cùng cấu hình\n\nCorrected/harmed dùng cùng participant so với baseline LightGBM. Matched controls không tham gia chọn nested winner; số tham số MLP và attention vẫn khác nhau.')
    code("display(read('paired_comparisons'))\ndisplay(read('matched_ablation'))")
    md('## Từng fold, cấu hình đã chọn và sensitivity theo seed\n\nSeeds43/44 giữ hyperparameters chọn bằng seed42; không phải cohort độc lập.')
    code("display(read('fold_results').query('seed==42').sort_values(['model','fold']))\ndisplay(read('selected_configurations'))\ndisplay(read('seed_sensitivity_summary'))")
    md('## Nuisance và thiếu tác vụ\n\nKhông chọn lại model từ stress. Translation/scale/pressure/time có thể làm mất thông tin hữu ích; không mặc định label-preserving. Neural dùng token masks; cây dùng Train medians.')
    code("display(read('robustness_summary'))\ndisplay(read('masking_summary'))")
    md('## Task/group attribution — held-out models\n\nIG giải thích positive-class logit, reference Train mean. TreeSHAP và IG khác reference; so sánh rank/share, không gán ý nghĩa nhân quả.')
    code("d=read('attribution_importance')\ndisplay(d.query(\"level=='task'\").sort_values(['model','rank']).groupby('model').head(5))\ndisplay(d.query(\"level=='group'\").sort_values(['model','rank']).groupby('model').head(5))")
    md('## Stability và faithfulness\n\nAttention là phân tích phụ. Completeness không chứng minh clinical validity. Permutation và zero-feature ablation là kiểm tra độc lập của ranking.')
    code("display(read('attribution_stability').groupby(['model','level'])[['spearman','top3_jaccard']].agg(['mean','std']).reset_index())\ndisplay(read('faithfulness'))\ndisplay(read('xai_baseline_rank_comparison'))\ndisplay(read('attention_agreement'))")
    md('## Learning diagnostics và chi phí\n\nNeural chạy GPU, LightGBM CPU; inference gồm preprocessing từ cached features, không gồm raw extraction. Gap mô tả tại inner best epoch.')
    code("display(read('overfitting_diagnostics').groupby('family')[['best_epoch','loss_gap','train_accuracy','validation_accuracy']].mean().reset_index())\ndisplay(read('inference_cost').groupby('model').median(numeric_only=True).reset_index())\ndisplay(pd.DataFrame([json.loads((OUT/'cache/validation.json').read_text())]))")
    md('## OOF và ca minh họa\n\nMọi người có đúng một dự đoán/method/seed. Ca local chọn ID đầu tiên theo từ điển cho từng outcome, chỉ để minh họa hậu nghiệm.')
    code("oof=read('oof_predictions')\ndisplay(oof.groupby(['model','seed']).agg(n=('participant_id','size'),unique_ids=('participant_id','nunique')).reset_index())\ndisplay(read('local_cases'))\ndisplay(read('paired_participant_changes').query(\"model=='transformer'\").head(20))")
    md('## Attribution từng participant\n\nXem cả corrected, harmed, đúng và sai. Signed values là đóng góp vào logit, không phải xác suất.')
    code("cases=read('local_cases'); a=read('participant_attributions')\nfor row in cases.itertuples():\n    display(Markdown(f'**{row.participant_id} — {row.change}**'))\n    display(a[(a.model=='transformer')&(a.participant_id==row.participant_id)].sort_values('absolute_mass',ascending=False))")
    md('## Hình phục vụ paper\n\nPNG300dpi, SVG và manifest lưu riêng. Chú thích chi tiết trong report; attention đặt supplementary.')
    code("manifest=json.loads((OUT/'figures/manifest.json').read_text())\nfor f in manifest:\n    display(Markdown('**'+f['name']+'** — '+f['caption']))\n    display(Image(filename=str(OUT/'figures'/(f['name']+'.png')),width=1000))")
    md('## Giới hạn và trạng thái\n\nTask1 đã xem cùng cohort/folds trước khi hình thành giả thuyết, nên train-only implementation không xóa research-level adaptive bias. Không có external/cross-language validation. Không coi ổn định XAI là đủ để bù accuracy thấp. Không dùng kết quả này làm chẩn đoán lâm sàng. Final `validation.json` được ghi sau khi notebook chạy và kiểm tra thành công.')
    code("display(Markdown('Trạng thái bàn giao cuối cùng: [validation.json](../results/task_transformer/validation.json). File được cập nhật sau khi notebook và unit tests vượt kiểm tra.'))\ndisplay(Markdown('Đọc kết luận đầy đủ tại [REPORT_VI.md](../results/task_transformer/REPORT_VI.md).'))")
    n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nbf.write(n,ROOT/'notebooks/06_kinematic_task_transformer.ipynb');print('NOTEBOOK CREATED')

if __name__=='__main__':main()
