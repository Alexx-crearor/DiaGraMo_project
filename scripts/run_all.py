"""Run the full study locally, with logs and restartable extraction/evaluation."""
import sys,subprocess,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--skip-download',action='store_true'); ap.add_argument('--device',default='auto'); args=ap.parse_args()
    logs=ROOT/'results/logs'; logs.mkdir(parents=True,exist_ok=True)
    history=[]; status=logs/'run_status.json'
    def run(name,arguments):
        print('START',name,flush=True); began=time.time()
        status.write_text(json.dumps(dict(current=name,completed=history),indent=2),encoding='utf-8')
        with (logs/(name+'.log')).open('a',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,'-X','utf8','-u',*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{name} failed; read results/logs/{name}.log')
        history.append(dict(stage=name,seconds=time.time()-began)); print('COMPLETE',name,flush=True)
    if not (ROOT/'data/derived/signals512.npy').exists(): run('prepare',['scripts/prepare.py'])
    if not args.skip_download: run('download_weights',['scripts/download_weights.py'])
    evaluation=None; eval_log=None
    def evaluate_async():
        nonlocal evaluation,eval_log
        if evaluation is not None:
            if evaluation.poll() is None: return
            if evaluation.returncode: raise RuntimeError('Evaluation failed; read results/logs/evaluate.log')
            eval_log.close()
        eval_log=(logs/'evaluate.log').open('a',encoding='utf-8')
        evaluation=subprocess.Popen([sys.executable,'-X','utf8','-u','scripts/evaluate.py'],cwd=ROOT,stdout=eval_log,stderr=subprocess.STDOUT)
    # Run encoders in an order that makes the published fusion available early.
    stages=[('moment',None),('gaf','vit_b_16'),('cwt','efficientnet_b4'),
            ('gaf','resnet50'),('cwt','resnet50'),('gaf','efficientnet_b4'),
            ('cwt','vit_b_16'),('gaf','vgg16'),('cwt','vgg16')]
    try:
        for rep,enc in stages:
            name=rep+('_'+enc if enc else '')
            cmd=['scripts/extract.py','--representation',rep,'--batch-size','8' if rep=='moment' else '16','--device',args.device]
            if enc: cmd+=['--encoder',enc]
            run(name,cmd); evaluate_async()
        if evaluation is not None:
            status.write_text(json.dumps(dict(current='evaluation',completed=history),indent=2),encoding='utf-8')
            code=evaluation.wait(); eval_log.close()
            if code: raise RuntimeError('Evaluation failed; read results/logs/evaluate.log')
        run('evaluate_final',['scripts/evaluate.py'])
        run('report',['scripts/report.py'])
        run('execute_notebooks',['scripts/execute_notebooks.py'])
        status.write_text(json.dumps(dict(current='complete',completed=history),indent=2),encoding='utf-8')
    except BaseException as exc:
        status.write_text(json.dumps(dict(current='failed',error=str(exc),completed=history),indent=2),encoding='utf-8')
        raise

if __name__=='__main__': main()
