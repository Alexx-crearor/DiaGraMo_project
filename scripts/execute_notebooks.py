"""Execute read/report notebooks with this project's Python interpreter."""
import sys,asyncio,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
def main():
    rows=[]
    for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
        n=nbformat.read(p,as_version=4)
        n=NotebookClient(n,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.validate(n)
        nbformat.write(n,p)
        errors=[o for c in n.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
        if errors: raise RuntimeError('Notebook has errors: '+p.name)
        rows.append(dict(notebook=p.name,executed_code_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in n.cells),errors=0))
        print('EXECUTED',p.name,flush=True)
    (ROOT/'results/notebook_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__': main()
