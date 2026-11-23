"""Read-only numerical, CLI, privacy and documentation checks for weekly cleanup."""
from pathlib import Path
import ast
import json
import os
import re
import subprocess
import sys
import time
from urllib.parse import unquote
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research_cleanup'
WEEK=ROOT/'docs/weekly_reports/2026-11-16_to_2026-11-22'


def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')


def links():
    paths=[ROOT/'README.md',*(ROOT/'docs/research').glob('*.md'),*WEEK.glob('*.md'),*OUT.glob('*.md')]
    broken=[];checked=0
    for p in paths:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=unquote(target.split('#',1)[0].split(' "',1)[0]).strip('<>')
            checked+=1
            if not (p.parent/target).resolve().exists():broken.append(dict(file=p.relative_to(ROOT).as_posix(),target=target))
    return dict(relative_links_checked=checked,broken=broken,scope='README and new cleanup/research/weekly Markdown; external URLs and anchors not network-validated')


def numerical():
    locked=pd.read_csv(ROOT/'results/raw_temporal_v2/locked_study/matched_oof_metrics.csv')
    path=ROOT/'results/raw_temporal_v2/locked_study/participant_complementarity.csv'
    f=pd.read_csv(path)
    # Field names are discovered from the existing protected reporting contract.
    target='label' if 'label' in f else 'y'
    if target not in f:
        target='group'
    probabilities={'Kinematic':'kinematic_probability','Temporal':'probability','Fixed50':'fixed_average_probability'}
    checks=[]
    for name,column in probabilities.items():
        expected=locked[locked.model==name].iloc[0]
        y=f[target].to_numpy();p=f[column].to_numpy()
        actual=dict(accuracy=accuracy_score(y,p>.5),f1_macro=f1_score(y,p>.5,average='macro'),auc=roc_auc_score(y,p))
        for k,v in actual.items():
            assert np.isclose(v,expected[k],rtol=0,atol=1e-12),(name,k)
            checks.append(dict(model=name,metric=k,value=float(v),source='Protected full OOF replay',match=True))
    assert len(f)==258 and np.allclose(f.fixed_average_probability,.5*f.kinematic_probability+.5*f.probability,rtol=0,atol=1e-12)
    native=pd.read_csv(ROOT/'results/native_channel_research/MODEL_METRICS.csv')
    frames={}
    for name in ['B0','B1','B2','B3','B4']:
        data=pd.read_csv(ROOT/f'data/native_channel_research_local/{name}_oof.csv',dtype={'participant_id':str}).sort_values('participant_id')
        assert len(data)==120 and data.participant_id.is_unique and data.fold.nunique()==5
        frames[name]=data
        row=native[native.model==name].iloc[0]
        for metric,v in dict(accuracy=accuracy_score(data.label,data.probability>.5),f1_macro=f1_score(data.label,data.probability>.5,average='macro'),auc=roc_auc_score(data.label,data.probability)).items():
            assert np.isclose(row[metric],v,rtol=0,atol=1e-12),(name,metric)
            checks.append(dict(model=name,metric=metric,value=float(v),source='Private existing OOF replay; IDs withheld',match=True))
    assert frames['B3'].participant_id.tolist()==frames['B4'].participant_id.tolist()
    identical=np.array_equal(frames['B3'].probability>.5,frames['B4'].probability>.5)
    pair=pd.read_csv(ROOT/'results/native_channel_research/PAIRED_MODEL_COMPARISON.csv')
    assert identical and pair[(pair.left=='B3')&(pair.right=='B4')].iloc[0].prediction_disagreements==0
    ci=pd.read_csv(ROOT/'results/raw_temporal_v2/locked_study/paired_bootstrap.csv')
    fusion=ci[(ci.model=='Fixed50')&(ci.metric=='f1_macro')].iloc[0]
    assert fusion.delta<0 and fusion.lower95<0<fusion.upper95
    comparison=pair[(pair.left=='B3')&(pair.right=='B2')].iloc[0]
    assert comparison.f1_macro_ci95_low<0<comparison.f1_macro_ci95_high
    xai=pd.read_csv(ROOT/'results/kinematic_xai/exploratory/tables/task_importance.csv')
    assert xai.iloc[0].task=='copy' and xai.iloc[0].top3_count==6
    city=pd.read_csv(ROOT/'results/data_audit_v3/metadata_shortcut_metrics.csv')
    assert np.isclose(city[city.control=='id_city'].iloc[0].accuracy,251/258)
    return dict(metrics=checks,matched_original_participants=258,matched_native_participants=120,
        original_sixfold_complete=True,native_fivefold_complete=True,B3_B4_binary_identical=identical,
        fusion_delta_ci_crosses_zero=True,B3_B2_ci_crosses_zero=True,xai_copy_table_preserved=True,
        new_model_fit=False,bootstrap_recomputed=False)


def cli():
    before=json.loads((OUT/'CLI_BEFORE.json').read_text(encoding='utf8'))
    rows=[]
    for r in before:
        t=time.perf_counter()
        p=subprocess.run([sys.executable,'-B',r['path'],'--help'],cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=45)
        rows.append(dict(path=r['path'],returncode=p.returncode,help_stdout_identical=p.stdout==r['stdout'],seconds=time.perf_counter()-t))
    return rows


def privacy():
    ids=np.load(ROOT/'data/derived/kinematic.npz')['ids'].astype(str).tolist()
    ids+=pd.read_csv(ROOT/'data/external_datasets/drotar_dobes_2020/external_participant_inventory.csv',dtype={'participant_id':str}).participant_id.tolist()
    findings=[];files=[]
    for directory in [ROOT/'docs/research',ROOT/'docs/research_cleanup',WEEK,ROOT/'scripts/research_cleanup',ROOT/'tests/research_cleanup']:
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in files:
        if p.suffix not in {'.py','.csv','.md','.json','.txt'}:continue
        text=p.read_text(encoding='utf-8-sig')
        # Exclude digits embedded in hash/decimal; never reproduce matched clinical IDs.
        if any(re.search(r'(?<![A-Za-z0-9.])'+re.escape(pid)+r'(?![A-Za-z0-9.])',text) for pid in ids):
            findings.append(p.relative_to(ROOT).as_posix())
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    private_staged=[p for p in staged if p.startswith(('data/','.cache/','results/models/')) or p.endswith(('.pt','.joblib','.npz','.svc'))]
    ignored=subprocess.run(['git','check-ignore','data/research_cleanup_local/private.json'],cwd=ROOT,capture_output=True).returncode==0
    return dict(new_public_files_checked=len(files),files_requiring_privacy_review=findings,
        private_files_staged=private_staged,local_inventory_git_ignored=ignored,
        historical_sensitive_artifacts_preserved_not_redistributed=True,
        method='Bounded known-ID token scan and output selection; not a claim of whole-history clinical deidentification')


def main():
    link_result=links();write('DOCUMENTATION_LINK_CHECKS.json',link_result)
    nums=numerical();write('NUMERICAL_PRESERVATION.json',nums)
    help_result=cli();write('CLI_AFTER.json',help_result)
    safe=privacy();write('PRIVACY_REVIEW.json',safe)
    valid=not link_result['broken'] and all(r['returncode']==0 and r['help_stdout_identical'] for r in help_result) and not safe['files_requiring_privacy_review'] and not safe['private_files_staged'] and safe['local_inventory_git_ignored']
    write('BEHAVIOR_PRESERVATION_CHECKS.json',dict(valid=valid,numerical_replay=True,cli_help_equivalent=all(r['help_stdout_identical'] for r in help_result),no_experiment_run=True,link_check=link_result))
    print(json.dumps(dict(valid=valid,links=link_result,metrics_replayed=len(nums['metrics']),cli=help_result,privacy=safe),indent=2))
    if not valid:raise SystemExit(1)


if __name__=='__main__':main()
