"""Create a read-only notebook: numerical results first, research figures next."""
import nbformat as nb
from kinematic_xai_common import ROOT


def main():
    cells = []
    def md(source):
        cells.append(nb.v4.new_markdown_cell(source))
    def code(source):
        cells.append(nb.v4.new_code_cell(source))
    md('''# 05. Kinematic Task-Level XAI

258 participants, 14 handwriting tasks, six saved outer folds, seed 42.
Notebook chỉ đọc artifact; **bảng số ở trước biểu đồ**, không fit lại mô hình.
Attribution raw ở log-odds; signed task sum và absolute attribution mass được tách riêng.

LOTO có huấn luyện lại và test-time median replacement là hai thí nghiệm khác nhau.
Các bảng outer-test là thăm dò, không dùng để lựa chọn kiến trúc rồi coi cùng Test là đánh giá mới độc lập.
''')
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
OUT = ROOT / 'results/kinematic_xai'
TABLES = OUT / 'exploratory/tables'
pd.set_option('display.max_columns', 40)
def table(name):
    return pd.read_csv(TABLES / (name + '.csv'))
def show(name):
    display(Image(filename=str(OUT / 'figures' / (name + '.png'))))
print('Kinematic XAI; source HEAD:', json.loads((OUT/'provenance.json').read_text())['head'])""")
    md('## Kết quả số cuối cùng\nMetric lưu dạng tỷ lệ; nhân 100 để đọc %. Delta là ablated − baseline. SD qua sáu fold không phải CI.')
    code("""ablation = table('task_ablation_summary')
display(ablation.loc[ablation.method=='original_baseline', ['method','accuracy','f1_macro','auc','balanced_accuracy','mean_f1_macro','std_f1_macro']])
display(ablation[ablation.method=='leave_one_task_out_refit'].sort_values('f1_macro',ascending=False))
display(ablation[ablation.method=='test_time_train_median'].sort_values('f1_macro',ascending=False))""")
    md('## A–C. Feature, Task và Source Group\nMass = mean(sum(abs(phi))). Các signed sums được báo riêng.')
    code("display(table('top20_features'))\ndisplay(table('task_importance'))\ndisplay(table('group_importance'))\ndisplay(table('task_signed_summary'))")
    code("display(table('task_group_matrix').pivot(index='task',columns='group',values='mean_absolute_shap_mass'))\ndisplay(table('task_importance_by_fold'))")
    md('## D–F. Kiểm chứng bằng permutation, ablation và stability\nPermutation có thể tạo tổ hợp out-of-distribution; 30 repeats/fold được trung bình trong fold trước khi tính fold SD.')
    code("display(table('permutation_importance'))\ndisplay(table('method_agreement'))\ndisplay(table('method_concordance'))\ndisplay(table('fold_stability_summary'))")
    code("display(table('participant_variability'))\ndisplay(table('correct_vs_incorrect_tasks'))\ndisplay(table('local_cases'))")
    md('## Thay đổi dự đoán từng participant\nGiữ cả corrected, harmed, unchanged_correct và unchanged_wrong.')
    code("display(table('task_ablation_oof').groupby(['method','task','change']).size().unstack(fill_value=0))\ndisplay(table('task_ablation_oof').head(28))\ndisplay(table('participant_task_attributions').head(28))")
    md('## Train-only artifacts riêng\nCác artifact dưới đây chỉ dùng outer Train. Khi tích hợp vào một inner CV mới, phải gọi lại API trên inner Train; không dùng outer-Train importance để chọn task trên inner Validation nằm trong đó.')
    code("display(pd.read_csv(OUT/'train_only/fold_task_importance.csv'))\nprint('Không có task subset/weight nào được chọn trong nghiên cứu này.')")
    md('## Biểu đồ feature/task/group')
    code("for name in ['shap_beeswarm','shap_bar','task_importance','feature_groups','task_group_heatmap','task_fold_ranks']:\n    show(name)")
    md('## Perturbation, retraining và sự bất đồng giữa các phương pháp')
    code("for name in ['permutation_task','permutation_group','task_ablation','method_rank_agreement','participant_rank_variability']:\n    show(name)")
    md('## Local explanations\nID đầu tiên theo thứ tự từ điển trong từng outcome baseline; lựa chọn hậu kiểm để minh họa. Waterfall cộng trong raw log-odds, p(group=1) chỉ là kết quả sigmoid sau khi cộng.')
    code("for name in ['waterfall_true_negative','waterfall_true_positive','waterfall_false_positive','waterfall_false_negative','task_signed_waterfall']:\n    show(name)")
    md('## Kiểm chứng số liệu và báo cáo\nValidation nghiên cứu tổng thể nằm ở `results/kinematic_xai/validation.json`.')
    code("validation=json.loads((OUT/'numeric_validation.json').read_text())\ndisplay(pd.DataFrame([validation['checks']]))\ndisplay(Markdown('[Đọc báo cáo nghiên cứu tiếng Việt](../results/kinematic_xai/REPORT_VI.md)'))")
    notebook = nb.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(display_name='Python (.venv)', language='python', name='python3')))
    target = ROOT / 'notebooks/05_kinematic_task_xai.ipynb'
    nb.write(notebook, target)
    print('CREATED', target)


if __name__ == '__main__':
    main()
