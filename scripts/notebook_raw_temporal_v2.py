"""Build and execute a read-only artifact notebook, never retrain on opening."""
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    notebook=nbf.v4.new_notebook();cells=[]
    def markdown(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    markdown('''# Raw Temporal Research Recovery — Phase 0–4

Notebook chỉ đọc artifacts đã lưu, **không huấn luyện mô hình**. Các bảng bên dưới là số liệu thực nghiệm đã hoàn tất tại thời điểm chạy cell. Phase còn thiếu được ghi rõ; không thay số liệu chưa chạy bằng0.

Tất cả kết quả nội bộ mới là **thăm dò** do cohort đã được nghiên cứu nhiều lần. Development chỉ dùng143 Train/72 validation trong outer Train0;43 outer Test không dùng để chọn thiết kế. Kinematic baseline và v1 được bảo toàn.''')
    code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if not (ROOT / 'results').exists(): ROOT = ROOT.parent
OUT = ROOT / 'results/raw_temporal_v2'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
def table(relative):
    path = OUT / relative
    if path.exists():
        frame = pd.read_csv(path)
        display(frame)
        return frame
    display(Markdown('**Chưa có artifact:** `' + relative + '`'))
def figure(name):
    path = OUT / 'figures' / (name + '.png')
    if path.exists(): display(Image(filename=str(path)))
    else: display(Markdown('Hình chưa được tạo: `' + name + '`'))
status = json.loads((OUT / 'validation.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'item': k, 'status': str(v)} for k, v in status.items()]))''')
    markdown('## Phase 0 — Kiểm tra kỹ thuật\nTiny Train memorization kiểm tra khả năng tối ưu, không đo generalization. Synthetic là fixture ở mức representation, không phải mô phỏng SVC vật lý hoàn chỉnh.')
    code("tiny = table('tables/phase0_memorization.csv')\nsynthetic = table('tables/phase0_synthetic_results.csv')")
    code("figure('p0_memorization_curves')\nfigure('p0_clipping_and_quantization')")
    markdown('## Phase 1 — Bảng số cuối mỗi cấu hình đã chạy\nCheckpoint chọn bằng development-validation BCE. Bảng này không phải OOF Nested CV. Nếu queue còn chạy, các cấu hình chưa hoàn tất sẽ không xuất hiện.')
    code('''rows = []
for path in sorted((OUT / 'phase1_bottleneck').glob('p1_*/result.json')):
    result = json.loads(path.read_text(encoding='utf-8'))
    if 'architecture' not in result: continue
    rows.append(dict(configuration=result['experiment_id'], **result['metrics'],
                     train_f1_macro=result['train_metrics']['f1_macro'],
                     best_epoch=result['best_epoch'], epochs=result['epochs_run'],
                     parameters=result['parameters'], seconds=result['training_seconds'],
                     peak_gpu_mib=result['peak_gpu_tensor_mib']))
development = pd.DataFrame(rows)
if len(development): display(development.sort_values('f1_macro', ascending=False).reset_index(drop=True))
else: display(Markdown('Chưa có cấu hình development hoàn tất.'))''')
    markdown('## Phase 1 — Probe vật lý\nRidge/RBF-Ridge dùng inner participant CV, scaler X/Y fit Train. Target thuộc đúng scope observation. Window/segment không có chuyển động hợp lệ bị loại complete-case, có bảng coverage. Feature map544 chiều, pooled32 chiều; multi4 probe128 chiều. Padding để lại dấu vết chiều dài; pen-up là input flag trực tiếp. Không diễn giải các probe như bằng chứng lâm sàng.')
    code("probes = table('tables/phase1_layer_probes.csv')\nif probes is not None:\n    display(probes[probes.method == 'ridge'].pivot(index='level', columns='target', values='r2').round(4))\ncoverage = table('tables/phase1_probe_coverage.csv')")
    code("retention = table('tables/phase1_pooling_physical_probes.csv')\nif retention is not None:\n    display(retention[retention.method == 'ridge'].pivot(index='pool', columns='target', values='r2').round(4))")
    code("figure('p1_controlled_comparison')\nfigure('p1_layer_information')\nfigure('p1_resource_cost')")
    markdown('## Phase 2 — Statistical, MIL và Stroke Transformer\nChỉ đọc kết quả khi có artifact. Không giữ Transformer chỉ vì kiến trúc mới.')
    code("phase2 = table('tables/phase2_development_comparison.csv')")
    markdown('## Phase 3 — Masked motor SSL\nPretraining chỉ được dùng participant Train của từng split. Không coi reconstruction tốt là cải thiện downstream khi chưa có đối chứng.')
    code("phase3 = table('tables/phase3_ssl_comparison.csv')")
    markdown('## Phase 4 — Nested CV và complementarity\nBảng development phía trên không thay thế bảng OOF. Fusion phải dùng cross-fitted Train inputs, không dùng outer Test labels để chọn trọng số.')
    code("nested = table('tables/phase4_nested_comparison.csv')\ncomplementarity = table('tables/phase4_complementarity.csv')")
    markdown('## Báo cáo và trạng thái xác minh\nMở báo cáo đã tồn tại. Thiếu báo cáo nghĩa là chưa hoàn thành, không phải kết quả âm tính.')
    code('''for name in ['PHASE0_REPORT_VI.md', 'PHASE1_REPORT_VI.md', 'PHASE2_REPORT_VI.md',
             'PHASE3_REPORT_VI.md', 'PHASE4_REPORT_VI.md', 'REPORT_VI.md', 'FINAL_DECISION_VI.md']:
    path = OUT / name
    display(Markdown(('Đã có: ' if path.exists() else 'Chưa hoàn thành: ') + '`' + name + '`'))
ledger = pd.read_csv(OUT / 'EXPERIMENT_LEDGER.csv')
display(ledger.groupby(['phase', 'training_status'], dropna=False).size().rename('experiments').reset_index())''')
    notebook.cells=cells
    notebook.metadata=dict(kernelspec=dict(display_name='Python 3 (DiaGraMo)',language='python',name='python3'),
        language_info=dict(name='python',version=sys.version.split()[0]))
    path=ROOT/'notebooks/07_raw_temporal_research_recovery.ipynb'
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(notebook,path)
    print('EXECUTED READ-ONLY NOTEBOOK',path,flush=True)


if __name__=='__main__':main()
