"""Execute a read-only notebook displaying the locked research evidence."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import sys
import os

ROOT=Path(__file__).resolve().parents[1]
cells=[nbformat.v4.new_markdown_cell('''# DiaGraMo — Locked Temporal Study

Notebook chỉ đọc artifacts; không train, fit calibration, chọn threshold hoặc fusion weights.
Kết quả là exploratory internal CV, không phải external validation. Xem source-label confounding trước khi diễn giải hiệu năng.'''),
nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
OUT=ROOT/'results/raw_temporal_v2/locked_study'
pd.set_option('display.max_columns',30)
pd.set_option('display.float_format',lambda x:f'{x:.6f}')
display(pd.DataFrame([json.loads((OUT/'status.json').read_text())]).drop(columns='inputs_sha256'))'''),
nbformat.v4.new_markdown_cell('## Kết quả cuối cùng trên đúng participant-matched OOF'),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'matched_oof_metrics.csv'))"),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'matched_fold_metrics.csv')); display(pd.read_csv(OUT/'fold_mean_sd.csv'))"),
nbformat.v4.new_markdown_cell('## Complementarity và bất định — không phải bằng chứng nhân quả'),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'fusion_effect_by_fold.csv')); display(pd.read_csv(OUT/'branch_error_overlap.csv')); display(pd.read_csv(OUT/'paired_bootstrap.csv'))"),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'participant_complementarity.csv').head(20))"),
nbformat.v4.new_markdown_cell('## Data validity — nguồn thu thập tương quan mạnh với group'),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'source_associations.csv')); display(pd.read_csv(OUT/'prefix_city_mapping.csv',keep_default_na=False)); display(pd.read_csv(OUT/'source_fold_overlap.csv'))"),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'descriptive_metrics_by_city.csv'))"),
nbformat.v4.new_markdown_cell('## Calibration diagnostics — không fit calibration trên OOF'),
nbformat.v4.new_code_cell("display(Image(filename=str(OUT/'calibration.png'))); display(pd.read_csv(OUT/'calibration_bins.csv'))"),
nbformat.v4.new_markdown_cell('## Ablation lịch sử: development, không phải six-fold OOF'),
nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'historical_ablation_evidence.csv'))"),
nbformat.v4.new_code_cell("display(Markdown((OUT/'LOCKED_RESEARCH_REPORT_VI.md').read_text(encoding='utf-8')))"),
]
book=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
client=NotebookClient(book,timeout=180,resources={'metadata':{'path':str(ROOT)}},kernel_name='python3')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
client.execute()
path=ROOT/'notebooks/09_locked_temporal_study.ipynb';nbformat.write(book,path)
assert not [o for c in book.cells for o in c.get('outputs',[]) if o.output_type=='error']
print('LOCKED ARTIFACT NOTEBOOK EXECUTED',path)
