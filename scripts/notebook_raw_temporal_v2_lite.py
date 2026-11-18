"""Execute an artifact-only notebook for budgeted recovery and lite validation."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    book=nbformat.v4.new_notebook();cells=[]
    def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):cells.append(nbformat.v4.new_code_cell(text))
    md('''# DiaGraMo — Phase 3 recovery và Phase 4 lite

Notebook chỉ đọc artifacts. Cấu hình khóa trước outer evaluation, một seed42 và sáu folds gốc. Phase 3 full SSL và nghiên cứu đầy đủ vẫn chưa hoàn tất. Các dòng partial OOF chỉ so sánh trên đúng participants đã có Raw predictions. Mọi kết quả là exploratory vì cohort đã được nghiên cứu nhiều lần.''')
    code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if not (ROOT / 'results').exists(): ROOT = ROOT.parent
OUT = ROOT / 'results/raw_temporal_v2'
LITE = OUT / 'phase4_lite'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
def table(name):
    path = LITE / name
    if path.exists(): display(pd.read_csv(path))
    else: display(Markdown('Chưa có artifact: `' + name + '`'))
display(pd.DataFrame([json.loads((LITE / 'status.json').read_text(encoding='utf-8'))]))''')
    md('## CUDA và epoch SSL nhỏ\nThời gian measured khác với ước lượng extrapolation. Reconstruction pilot không chứng minh giá trị downstream.')
    code('''for name in ['cuda_smoke.json', 'pilot_result.json']:
    path = OUT / 'phase3_recovery' / name
    if path.exists(): display(pd.DataFrame([json.loads(path.read_text(encoding='utf-8'))]))''')
    md('## Bảng số OOF — Raw, Kinematic và 50/50\nKinematic trong từng candidate được giới hạn đúng cùng IDs/folds. Chỉ full258 người khi đủ sáu folds.')
    code("table('paired_comparison.csv')\ntable('fold_metrics.csv')")
    md('## Paired error overlap và bootstrap\nCorrected cases riêng lẻ không chứng minh complementary value. Bootstrap conditional trên models đã fit.')
    code("table('error_overlap.csv')\ntable('paired_deltas.csv')")
    code('''path = LITE / 'figures/paired_f1.png'
if path.exists(): display(Image(filename=str(path)))''')
    md('## Thời gian, bộ nhớ và checkpoint\nAllocated/reserved tensor memory khác với tổng process memory. Không retrain khi mở notebook.')
    code("table('runtime_memory.csv')\ntable('validation_checks.csv')")
    code('''rows = []
for path in sorted((LITE / 'fits').glob('*/status.json')):
    rows.append(dict(fit=path.parent.name, **json.loads(path.read_text(encoding='utf-8'))))
display(pd.DataFrame(rows))
for name in ['PHASE3_RECOVERY_VI.md', 'PHASE4_LITE_REPORT_VI.md']:
    path = OUT / name
    if path.exists(): display(Markdown(path.read_text(encoding='utf-8')))''')
    book.cells=cells;book.metadata=dict(kernelspec=dict(display_name='Python 3 (DiaGraMo)',language='python',name='python3'))
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    path=ROOT/'notebooks/08_raw_temporal_lite.ipynb';nbformat.write(book,path)
    print('EXECUTED ARTIFACT-ONLY LITE NOTEBOOK',path,flush=True)


if __name__=='__main__':main()
