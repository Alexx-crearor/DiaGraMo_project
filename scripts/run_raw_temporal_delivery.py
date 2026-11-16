"""Finish reports after genuine fit/XAI completion; final visual review stays explicit."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw_temporal'
_last_progress=None
def progress(stage):
    global _last_progress
    inner=sum(len(list((OUT/'folds'/f'fold_{f}'/'inner').glob('*.oof.csv')))for f in range(6))
    outer=sum(len(list((OUT/'folds'/f'fold_{f}'/'outer').glob('*.oof.csv')))for f in range(6))
    training=[f for f in range(6)if(OUT/'folds'/f'fold_{f}'/'complete.json').exists()]
    xai=[f for f in range(6)if(OUT/'xai'/f'fold_{f}'/'complete.json').exists()]
    state=dict(stage=stage,complete=False,inner_fits_completed=inner,inner_fits_expected=216,
        outer_fits_completed=outer,outer_fits_expected=96,training_folds_completed=training,xai_folds_completed=xai,
        pending_training_folds=sorted(set(range(6))-set(training)),pending_xai_folds=sorted(set(range(6))-set(xai)),
        final_validation='pending; consult validation.json for whole-study completion')
    temporary=OUT/'delivery_progress.tmp'
    for attempt in range(6):
        try:
            temporary.write_text(json.dumps(state,indent=2),encoding='utf-8');temporary.replace(OUT/'delivery_progress.json');break
        except PermissionError:time.sleep(.1)
    else:print('Progress file temporarily locked; continue computation and retry next update',flush=True)
    current=(stage,inner,outer,tuple(xai))
    if current!=_last_progress:print(stage,inner,'/216 inner',outer,'/96 outer','XAI folds',xai,flush=True);_last_progress=current
def run(script,*args):
    progress('RUN '+script+' '+' '.join(args))
    with (OUT/'delivery.log').open('a',encoding='utf-8')as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-B','scripts/'+script,*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        result=p.wait()
    if result:
        (OUT/'validation.json').write_text(json.dumps(dict(complete=False,stage='failed_'+script,returncode=result),indent=2),encoding='utf-8')
        raise SystemExit(result)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-from',choices=['xai_validation']);args=parser.parse_args()
    progress('waiting_for_all_six_training_and_cpu_xai_folds')
    while not all((OUT/'folds'/f'fold_{f}'/'complete.json').exists()and(OUT/'xai'/f'fold_{f}'/'complete.json').exists()for f in range(6)):
        time.sleep(5);progress('waiting_for_all_six_training_and_cpu_xai_folds')
    if args.resume_from is None:
        run('evaluate_raw_temporal.py')
        run('analyze_raw_temporal_xai.py','--examples-only','--device','cpu')
        run('summarize_raw_temporal_xai.py')
        run('validate_raw_temporal.py','--phase','fits')
    else:
        assert json.loads((OUT/'fit_validation.json').read_text())['complete']
        assert json.loads((OUT/'xai/summary_complete.json').read_text())['complete']
    run('validate_raw_temporal.py','--phase','xai')
    run('report_raw_temporal_audit.py')
    run('report_raw_temporal.py')
    run('plot_raw_temporal.py')
    run('create_raw_temporal_notebook.py')
    for name in ('raw_temporal_jupyter','raw_temporal_ipython'):(ROOT/'tmp'/name).mkdir(parents=True,exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'tmp/raw_temporal_jupyter');os.environ['IPYTHONDIR']=str(ROOT/'tmp/raw_temporal_ipython');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    run('execute_notebooks.py','--only','06_raw_temporal_graphomotor.ipynb','--validation-output','results/raw_temporal/notebook_validation.json')
    progress('ready_for_visual_review_and_final_validation')

if __name__=='__main__':main()
