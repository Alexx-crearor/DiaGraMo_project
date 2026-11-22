"""Only the admitted prior control. No raw models on unverified signal semantics."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score,f1_score,roc_auc_score,balanced_accuracy_score,
                             confusion_matrix,brier_score_loss,log_loss)


def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if y.shape!=p.shape or len(y)==0 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probability inputs')
    if not set(np.unique(y))<={0,1}:raise ValueError('Nonbinary labels')
    pred=(p>.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    result=dict(accuracy=float(accuracy_score(y,pred)),f1_macro=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
        auc=float(roc_auc_score(y,p)) if both else None,
        balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
        sensitivity=float(tp/(tp+fn)) if tp+fn else None,specificity=float(tn/(tn+fp)) if tn+fp else None,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    bins=np.minimum((p*10).astype(int),9);ece=0.
    for k in range(10):
        m=bins==k
        if m.any():ece+=m.mean()*abs(y[m].mean()-p[m].mean())
    result['ece10']=float(ece)
    return result


def context(n,folds):return dict(dataset='Drotar_Dobes_2020',participant_count=n,target_definition='Workbook DYSGR vs0; study assessment',
    split_protocol='Participant Stratified5 shuffle seed42',model='Train_prior_label_control',completed_folds=folds,
    evaluation_scope='Exploratory external-dataset-only internal CV; label-only control')


def run(c,budget):
    import psutil
    import hashlib
    start=time.monotonic();cpu_start=time.process_time();peak_rss=psutil.Process().memory_info().rss
    cfg=c.load('LOCKED_CONFIG.json');digest=c.sha(c.OUT/'LOCKED_CONFIG.json')
    if cfg['allowed_experiments']!=['Train_prior_label_control'] or cfg['signal_experiments_allowed']:
        raise ValueError('Unrecognized locked experiment; no arbitrary training dispatch')
    assert digest==c.load('PREEXPERIMENT_DECISIONS.json')['locked_config_sha256']
    participants=pd.read_csv(c.LOCAL/'external_participant_inventory.csv',dtype={'participant_id':str},keep_default_na=False)
    assert participants.participant_id.is_unique and len(participants)==120
    source=c.load('external_dataset_checksums.json')
    for item in source['files']:
        if c.sha(c.LOCAL/item['name'])!=item['sha256']:raise ValueError('Dataset checksum changed')
    oof=[];foldrows=[];runrows=[];checkpoints=[];durations=[]
    for fold in range(5):
        tr=participants[participants.candidate_fold!=fold];te=participants[participants.candidate_fold==fold]
        assert not set(tr.participant_id)&set(te.participant_id) and tr.label.nunique()==2 and te.label.nunique()==2
        split_sha=hashlib.sha256(json.dumps(dict(train=tr.participant_id.tolist(),test=te.participant_id.tolist()),sort_keys=True).encode()).hexdigest()
        path=c.LOCAL/f'checkpoints/prior_fold{fold}.json';manifestpath=c.LOCAL/f'checkpoints/prior_fold{fold}.sha256.json'
        t=time.monotonic();cpu=time.process_time()
        estimated=max(1.,max(durations,default=1.)*2)
        budget.admit(estimated)
        if path.exists():
            if not manifestpath.exists():raise ValueError('Incomplete checkpoint cannot be trusted')
            recorded=json.loads(manifestpath.read_text(encoding='utf-8'))
            if c.sha(path)!=recorded['sha256']:raise ValueError('Checkpoint hash mismatch')
            state=json.loads(path.read_text(encoding='utf-8'))
            if state['config_sha256']!=digest or state['split_sha256']!=split_sha or state['fold']!=fold:raise ValueError('Model/fold/signature mismatch')
            expected=[int((tr.label==k).sum()) for k in (0,1)]
            if state['class_counts']!=expected:raise ValueError('Checkpoint Train class counts differ')
            probability=state['probability'];action='resumed_no_refit'
            if probability!=expected[1]/sum(expected):raise ValueError('Checkpoint probability replay differs')
        else:
            model=DummyClassifier(strategy='prior',random_state=42)
            model.fit(np.zeros((len(tr),1)),tr.label.to_numpy())
            p=model.predict_proba(np.zeros((len(te),1)))[:,list(model.classes_).index(1)]
            probability=float(p[0]);counts=[int((tr.label==k).sum()) for k in (0,1)]
            assert np.array_equal(p,np.full(len(te),counts[1]/sum(counts)))
            state=dict(fold=fold,config_sha256=digest,split_sha256=split_sha,class_counts=counts,probability=probability,
                train_ids=tr.participant_id.tolist(),test_ids=te.participant_id.tolist(),
                model='closed-form Train prior; no neural parameters',complete=True)
            c.write(f'checkpoints/prior_fold{fold}.json',state,local=True)
            c.write(f'checkpoints/prior_fold{fold}.sha256.json',dict(sha256=c.sha(path)),local=True)
            action='fit_completed'
        p=np.full(len(te),probability);score=metrics(te.label,p)
        duration=time.monotonic()-t;durations.append(duration)
        foldrows.append(dict(**context(len(te),1),fold=fold,**score,confidence_interval='N/A individual fold',metric='Fold metrics'))
        runrows.append(dict(**context(len(te),1),fold=fold,wall_seconds=duration,cpu_seconds=time.process_time()-cpu,
            process_rss_bytes=psutil.Process().memory_info().rss,gpu_allocated_bytes=0,gpu_reserved_bytes=0,gpu_training_seconds=0,
            status=action,metric='Runtime',confidence_interval='N/A',forecast_next_fold_seconds=estimated))
        for pid,y,prob in zip(te.participant_id,te.label,p):oof.append(dict(participant_id=pid,label=int(y),fold=fold,probability=float(prob),prediction=int(prob>.5)))
        checkpoints.append(dict(alias=f'prior_fold{fold}',location=f'local-only data/external_datasets/.../checkpoints/prior_fold{fold}.json',
            sha256=c.sha(path),split_sha256=split_sha,config_sha256=digest,complete=True,optimizer='N/A closed-form',
            replay_exact=True))
        # Persist completed-fold evidence immediately; interruption never manufactures missing coverage.
        c.table('prior_oof.csv',oof,local=True)
        c.table('fold_metrics.csv',foldrows)
        c.table('runtime_memory.csv',runrows)
        c.write('checkpoint_manifest.json',dict(checkpoints=checkpoints))
        c.write('EXPERIMENT_PROGRESS.json',dict(completed_folds=len(foldrows),expected_folds=5,coverage=len(oof),config_sha256=digest))
        peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
    frame=pd.DataFrame(oof).sort_values('participant_id')
    if len(frame)!=120 or not frame.participant_id.is_unique or set(frame.fold)!=set(range(5)):raise ValueError('Incomplete OOF excluded from whole-cohort metrics')
    score=metrics(frame.label,frame.probability)
    # Conditional participant bootstrap. No site labels exist for cluster resampling.
    rng=np.random.default_rng(42);y=frame.label.to_numpy();p=frame.probability.to_numpy()
    boot={k:[] for k in ['accuracy','f1_macro','auc','balanced_accuracy','sensitivity','specificity','brier','log_loss','ece10']}
    budget.admit(30)
    for _ in range(cfg['uncertainty']['bootstrap_replicates']):
        at=rng.integers(0,len(frame),len(frame));m=metrics(y[at],p[at])
        for k in boot:
            if m[k] is not None:boot[k].append(m[k])
    rows=[]
    for metric,value in score.items():
        values=boot.get(metric)
        lo,hi=map(float,np.quantile(values,[.025,.975])) if values else (None,None)
        rows.append(dict(**context(120,5),metric=metric,value=value,ci95_low=lo,ci95_high=hi,
            confidence_interval='2000IID participant OOF bootstrap; conditional, no source clusters' if values else 'N/A confusion count',
            fold_mean=float(np.mean([r[metric] for r in foldrows])) if metric in boot else None,
            fold_sd=float(np.std([r[metric] for r in foldrows],ddof=1)) if metric in boot else None))
    c.table('model_metrics.csv',rows)
    bins=np.minimum((p*10).astype(int),9);cal=[]
    for b in range(10):
        mask=bins==b
        cal.append(dict(**context(120,5),bin=b,n=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,
            positive_fraction=float(y[mask].mean()) if mask.any() else None,metric='Fixed10-bin calibration',confidence_interval='Not estimated per bin'))
    c.table('calibration_bins.csv',cal)
    c.table('experiment_registry.csv',[
        dict(**context(120,5),path='D-prior-only',status='COMPLETE',reason='Admitted label-balance diagnostic; no raw input',metric='See model_metrics.csv',confidence_interval='See model_metrics.csv'),
        *[dict(dataset='Drotar_Dobes_2020',participant_count=120,target_definition=cfg['target'],split_protocol='NOT_RUN',model=model,
            completed_folds=0,evaluation_scope=scope,path=path,status=status,reason=reason,metric='NOT_COMPUTED',confidence_interval='N/A')
            for path,model,scope,status,reason in [
                ('A','Frozen Kinematic/Stroke/Fixed50','Exact frozen external transfer','INVALID','14-task/physical schema contract fails'),
                ('B','Shared-task models','Shared-task transfer','NOT_FEASIBLE','No executable verified task mapping'),
                ('C','Task-agnostic raw models','Task-agnostic exploratory experiment','BLOCKED','Unverified export semantics/units'),
                ('D-signals','Handcrafted Kinematic/compact Temporal','External-dataset-only internal CV','BLOCKED','No verified signal mapping/units; no native-unit guessing')]]])
    validation=dict(prior_control_complete=True,completed_folds=5,participants=120,config_sha256=digest,
        no_train_test_id_overlap=True,all_repeated_recordings_grouped=True,train_only_class_counts=True,
        checkpoint_hash_and_replay_verified=True,original_checkpoints_loaded=False,signal_models_fit=0,
        fusion_computed=False,no_test_model_selection=True,source_independence_verified=False,
        full_requested_signal_comparison_complete=False,wall_seconds=time.monotonic()-start,
        cpu_seconds=time.process_time()-cpu_start,peak_observed_process_rss_bytes=peak_rss,
        gpu_training_seconds=0,gpu_allocated_bytes=0,gpu_reserved_bytes=0,
        memory_method='RSS sampled at stage/fold boundaries, not OS-wide peak; CUDA tensor allocation unused')
    c.write('experiment_validation.json',validation)
    c.md('PHASE6_EXPERIMENT_REPORT.md',f'''# Phase 6 — Thí nghiệm thực sự đã chạy

Chỉ chạy **Train-prior label control**,5/5folds,120uniqueIDs,participantStratified5/seed42, Trainclasscounts, fixedthresholdp>0.5. Đây là diagnosticlabelbalance trongexternaldataset, **không phải Kinematic, Temporal hoặc externaltransfer**.

OOFAccuracy={score['accuracy']:.6f}, MacroF1={score['f1_macro']:.6f}, AUC={score['auc']:.6f}, BalancedAccuracy={score['balanced_accuracy']:.6f}, Sensitivity={score['sensitivity']:.6f}, Specificity={score['specificity']:.6f}. Các metric/CI và foldmean±SD thực nằm ở model_metrics.csv/fold_metrics.csv. AUCpriorOOF có thể khác0.5 vì Trainprior thay nhẹ ngược với tỷ lệheldoutfold; không coi là thông tin motor.

Checkpoint5folds là JSONsufficientstatistics, classcounts/splitIDs và recipehash; hashverified/replayexact; optimizer/RNGstate N/A vìclosedform. OOF và exactsplitIDs giữlocal-only. Source/time-windowmodels không fit, không loadoldcheckpoints, không dùngrawcột chưa định nghĩa. Fusion/complementarityKinematicTemporal NOT_COMPUTED. A INVALID; B NOTFEASIBLE; C vàDsignalmodels BLOCKED. **PARTIAL** đối với suite nghiên cứu, dùcontrol đã hoàn tất.

Runtimecontrol+metrics/bootstrap={validation['wall_seconds']:.3f}s wall/{validation['cpu_seconds']:.3f}s processCPU; observedRSSmax={peak_rss/2**20:.1f}MiB (sampled, khôngOSpeak); GPUtraining/tensorallocation=0. BaselineCUDAcontext/resourceprobe không được gọi là GPUtraining.
''')
    c.status(6,'PARTIAL',start,dict(prior_metrics=score,completed_folds=5,signal_models=0),validation,
        ['PHASE6_EXPERIMENT_REPORT.md','experiment_registry.csv','model_metrics.csv','fold_metrics.csv','runtime_memory.csv','checkpoint_manifest.json','experiment_validation.json'],
        reason='Only admitted label-control finished; signal-based learning/transfer not scientifically admissible under current schema evidence')
    return validation
