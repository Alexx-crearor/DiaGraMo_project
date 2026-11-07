"""Execute read/report notebooks with this project's Python interpreter."""
import sys,asyncio,json,argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+',help='Notebook filenames in notebooks/; others are preserved')
    parser.add_argument('--validation-output',help='Results JSON path; selective runs use a separate default')
    args=parser.parse_args()
    paths=sorted((ROOT/'notebooks').glob('*.ipynb'))
    if args.only:
        missing=set(args.only)-{p.name for p in paths}
        if missing: raise ValueError('Unknown notebooks: '+str(sorted(missing)))
        paths=[p for p in paths if p.name in args.only]
    destination=args.validation_output or ('results/notebook_validation_selected.json' if args.only
                                           else 'results/notebook_validation.json')
    target=(ROOT/destination).resolve()
    if not target.is_relative_to(ROOT/'results'): raise ValueError('Validation must be saved inside results/')
    rows=[]
    for p in paths:
        n=nbformat.read(p,as_version=4)
        n=NotebookClient(n,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.validate(n)
        nbformat.write(n,p)
        errors=[o for c in n.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
        if errors: raise RuntimeError('Notebook has errors: '+p.name)
        rows.append(dict(notebook=p.name,executed_code_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in n.cells),errors=0))
        print('EXECUTED',p.name,flush=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__': main()
