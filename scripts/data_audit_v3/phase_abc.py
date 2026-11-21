"""Reconstruct cohort and labels; audit source associations and a frozen negative control."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import re
import time
from typing import Any
from scipy.stats import chi2_contingency
from common import *

SOURCE_FIELDS=['id_city','id_school_ppp','site_type_ppp','date_of_administration']


def label_table(cohort: pd.DataFrame) -> pd.DataFrame:
    components=[field for field in cohort if field.endswith('flag_to_omit')]
    labels=cohort[['participant_id','group','diagnosis','flag_to_omit_total']+SOURCE_FIELDS+components].copy()
    labels['diagnosis_missing']=missing(labels.diagnosis)
    numeric_group=pd.to_numeric(labels.group,errors='coerce');numeric_diagnosis=pd.to_numeric(labels.diagnosis,errors='coerce')
    labels['label_disagreement']=(numeric_group!=numeric_diagnosis)&numeric_group.notna()&numeric_diagnosis.notna()
    labels['omit_flag']=pd.to_numeric(labels.flag_to_omit_total)==1
    numeric=labels[components].apply(pd.to_numeric,errors='coerce')
    labels['observed_flag_components']=''
    for index in labels.index:labels.loc[index,'observed_flag_components']='|'.join(field for field in components if numeric.loc[index,field]==1)
    labels['missing_component_flags']=numeric.isna().sum(1)
    labels['component_or_matches_total']=(numeric.eq(1).any(axis=1)==labels.omit_flag).where(~numeric.isna().any(axis=1),None)
    labels['reason_status']='Observed component field names only; clinical meaning and official omission-generation rule not verified'
    return labels


def identity_links(people: pd.DataFrame,links: pd.DataFrame) -> None:
    frame=people.merge(links[['participant_id','has_original_svc','has_raw_cognitive_id']],on='participant_id',validate='one_to_one')
    frame['metadata_id_present']=~missing(frame.metadata_participant_id)
    anomaly=frame[(~frame.metadata_id_matches_filename)|(~frame.has_original_svc)|(~frame.has_raw_cognitive_id)].copy()
    def syntactic(value: str) -> str:
        match=re.fullmatch(r'([A-Za-z]+)(\d+)',value)
        return match[1]+str(int(match[2])) if match else value
    anomaly['numeric_suffix_normalized_key']=anomaly.participant_id.map(syntactic)
    anomaly['interpretation']='ID-string correspondence diagnostic only; not a verified person linkage; no labels reassigned'
    csv('identity_link_anomalies.csv',anomaly)


def category(values: pd.Series) -> pd.Series:
    return values.where(~missing(values),'<missing>').astype(str)


def source_majority(train_source: np.ndarray,train_y: np.ndarray,test_source: np.ndarray,strength: float=2.) -> np.ndarray:
    """Train-only beta-prior shrinkage; unseen source falls back to Train prior."""
    train_y=np.asarray(train_y,dtype=int)
    if len(train_source)!=len(train_y) or len(train_y)==0:raise ValueError('Invalid negative-control Train data')
    prior=(train_y.sum()+1)/(len(train_y)+2)
    mapping={value:(train_y[train_source==value].sum()+strength*prior)/((train_source==value).sum()+strength) for value in np.unique(train_source)}
    return np.asarray([mapping.get(value,prior) for value in test_source],dtype=float)


def association(values: pd.Series,y: np.ndarray,permutations: int=999) -> tuple[dict[str,Any],pd.DataFrame]:
    labels=category(values);table=pd.crosstab(labels,np.asarray(y)).reindex(columns=[0,1],fill_value=0)
    array=table.to_numpy();n=int(array.sum());r=len(table)
    chi,p,_,expected=chi2_contingency(array,correction=False) if r>1 else (0.,1.,0,np.asarray(array,dtype=float))
    codes=pd.Categorical(labels,categories=table.index).codes
    def statistic(test_y: np.ndarray) -> float:
        counts=np.bincount(2*codes+test_y,minlength=2*r).reshape(r,2)
        return float(((counts-expected)**2/np.maximum(expected,1e-100)).sum())
    rng=np.random.default_rng(SEED);more=sum(statistic(rng.permutation(y))>=chi-1e-10 for _ in range(permutations))
    return dict(n=n,categories=r,mixed_sources=int((array>0).sum(1).tolist().count(2)),
        single_class_sources=int(((array>0).sum(1)==1).sum()),label_purity=float(array.max(1).sum()/n),
        cramers_v=float(np.sqrt(chi/n)) if r>1 else 0.,chi_square=float(chi),chi_square_asymptotic_p=float(p),
        expected_cells_lt5=int((expected<5).sum()),expected_cells=int(expected.size),
        sparse_table=bool((expected<5).any()),permutation_p=(more+1)/(permutations+1),permutations=permutations,
        inference='Participant exchangeability association diagnostic, not causal; clustering/source selection not eliminated'),table


def main() -> None:
    started=time.perf_counter();assignment=folds();selected=set(assignment.participant_id)
    raw_path=ROOT/'data/raw_data/cognitive_test_data/cognitiveTestCSV2data.csv'
    raw=pd.read_csv(raw_path,sep=None,engine='python',keep_default_na=False)
    dictionary=[]
    def describe_metadata(frame: pd.DataFrame,source: str) -> None:
        for field in frame:
            miss=missing(frame[field]);name=field.lower()
            domain='clinical_or_screening' if any(v in name for v in ['group','diagnos','care_type','flag_to_omit']) else ('source_or_covariate' if any(v in name for v in ['city','school','site','date','age','sex','language','birth','class','device','protocol','batch','recruit']) else 'measurement_or_other')
            dictionary.append(dict(field=field,source=source,dtype=str(frame[field].dtype),rows=len(frame),missing=int(miss.sum()),
                cardinality=int(frame.loc[~miss,field].nunique()),domain=domain,meaning_status='Field name/data presence verified; clinical definitions not independently confirmed'))
    describe_metadata(raw,raw_path.relative_to(ROOT).as_posix())
    duplicates=[]
    for pid,part in raw.groupby('id_participant',dropna=False):
        if len(part)>1:duplicates.append(dict(participant_id=pid,rows=len(part),
            conflicting_fields='|'.join(c for c in raw if part[c].nunique(dropna=False)>1),exact_rows_identical=len(part.drop_duplicates())==1))
    csv('duplicate_metadata_ids.csv',pd.DataFrame(duplicates,columns=['participant_id','rows','conflicting_fields','exact_rows_identical']))
    people=[];recordings=[];metadata=[];source_hashes=[]
    clean=sorted((ROOT/'data/clean_data').glob('*_merged.json'))
    for number,path in enumerate(clean):
        content=path.read_bytes();data=loads(content);pid=path.stem.removesuffix('_merged')
        metas=[value[0] for key,value in data.items() if key.endswith('_cognitive_test_data') and value]
        if len(metas)>1:raise ValueError('Multiple cognitive records in merged source')
        meta=metas[0] if metas else {};metadata.append(meta)
        found=defaultdict(list);issues=[]
        for key,record in data.items():
            task=canonical(key)
            if task not in TASKS:continue
            a=record['data'];lengths={c:len(a.get(c,[])) for c in RAW_ORDER}
            finite=sum(int((~np.isfinite(np.asarray(a.get(c,[]),dtype=float))).sum()) for c in RAW_ORDER)
            valid=len(set(lengths.values()))==1 and min(lengths.values())>=4
            if not valid or finite:issues.append(task+':invalid_signals')
            row=dict(participant_id=pid,task=task,source_key=key,samples=lengths['x'],
                equal_channel_lengths=valid,nonfinite=finite,included=pid in selected,
                raw_source=(Path('data/raw_data/handwriting_data/data_SVC')/pid/(key[:-5]+'.svc')).as_posix())
            for channel in RAW_ORDER:
                values=np.asarray(a.get(channel,[]),dtype=float);row[channel+'_sha256']=array_sha(values)
                if channel=='pressure':row['pressure_round4_sha256']=array_sha(np.round(values,4))
            recordings.append(row);found[task].append(a)
        missing_tasks=[task for task in TASKS if task not in found];duplicate_tasks=[task for task,items in found.items() if len(items)>1]
        group=meta.get('group');valid_label=group in (0,1)
        included=not missing_tasks and not duplicate_tasks and not issues and valid_label
        record=dict(participant_id=pid,metadata_participant_id=meta.get('id_participant'),metadata_id_matches_filename=meta.get('id_participant')==pid,
            group=group,diagnosis=meta.get('special_care_type_dysgraphia'),
            diagnosis_alias_source='special_care_type_dysgraphia; not an independent clinical label',
            flag_to_omit_total=meta.get('flag_to_omit_total'),task_count=len(found),recording_count=sum(map(len,found.values())),
            complete_14=not missing_tasks,no_duplicate_tasks=not duplicate_tasks,signals_valid=not issues,
            valid_label=valid_label,included=included,missing_tasks='|'.join(missing_tasks),duplicate_tasks='|'.join(duplicate_tasks),issues='|'.join(issues),
            **{field:meta.get(field) for field in SOURCE_FIELDS+['age','sex_female','class','language','country_of_birth']},
            **{field:meta.get(field) for field in meta if field.endswith('flag_to_omit')},
            alphabetic_prefix=re.match(r'^[A-Za-z]+',pid).group())
        people.append(record)
        source_hashes.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content)))
        if number%40==0:print('PHASE A MERGED SOURCES',number+1,'/',len(clean),flush=True)
    people=pd.DataFrame(people);recordings=pd.DataFrame(recordings);meta_frame=pd.DataFrame(metadata)
    describe_metadata(meta_frame,'data/clean_data/*_merged.json::_cognitive_test_data')
    actual=people[people.included].set_index('participant_id').loc[assignment.participant_id]
    if not actual.metadata_id_matches_filename.all():raise AssertionError('Selected participant filename/metadata ID mismatch')
    if set(people.loc[people.included,'participant_id'])!=selected:raise AssertionError('Reconstructed cohort differs from protected cohort')
    np.testing.assert_array_equal(actual.group.astype(int),assignment.label)
    actual=actual.reset_index().merge(assignment,on='participant_id',validate='one_to_one')
    raw_core=['group','special_care_type_dysgraphia','flag_to_omit_total','id_city','id_schoolPPP']
    joins=[]
    for pid,part in raw.groupby('id_participant'):
        if pid not in set(people.participant_id):continue
        reference=people.set_index('participant_id').loc[pid]
        for field,target in zip(raw_core,['group','diagnosis','flag_to_omit_total','id_city','id_school_ppp']):
            if field not in part:
                joins.append(dict(participant_id=pid,field=field,raw_nonmissing_distinct=0,matches_merged=None,verification='UNVERIFIABLE: field absent from raw CSV'))
                continue
            vals=part[field].replace('',np.nan).dropna().astype(str).unique();expected=reference[target]
            numeric=field in raw_core[:3]
            matched=all(float(v)==float(expected) for v in vals) if numeric and pd.notna(expected) else (all(v==str(expected) for v in vals) if pd.notna(expected) else len(vals)==0)
            joins.append(dict(participant_id=pid,field=field,raw_nonmissing_distinct=len(vals),matches_merged=matched,verification='Compared all duplicate-row values'))
    join=pd.DataFrame(joins);csv('metadata_join_checks.csv',join)
    # Contradictions are recorded, not silently repaired. Current cohort join must match.
    compared=join[join.participant_id.isin(selected)&join.verification.eq('Compared all duplicate-row values')]
    if not compared.matches_merged.eq(True).all():raise AssertionError('Selected-cohort label/source mismatch')
    csv('participants_inventory.csv',people);csv('merged_recording_inventory.csv',recordings)
    csv('audit_cohort.csv',actual);csv('metadata_dictionary.csv',pd.DataFrame(dictionary))
    flow=[];remaining=np.ones(len(people),dtype=bool)
    for stage,condition in [('all_merged',np.ones(len(people),bool)),('complete_14_tasks',people.complete_14),
        ('no_duplicate_tasks',people.no_duplicate_tasks),('valid_signal_arrays',people.signals_valid),('binary_group',people.valid_label)]:
        before=int(remaining.sum());remaining&=np.asarray(condition);flow.append(dict(stage=stage,entering=before,retained=int(remaining.sum()),excluded_at_step=before-int(remaining.sum())))
    csv('cohort_selection_flow.csv',pd.DataFrame(flow))
    raw_svc=list((ROOT/'data/raw_data/handwriting_data/data_SVC').rglob('*.svc'))
    raw_ids={p.parent.name for p in raw_svc}
    links=people[['participant_id','metadata_participant_id','metadata_id_matches_filename','included']].copy()
    links['has_original_svc']=links.participant_id.isin(raw_ids);links['has_raw_cognitive_id']=links.participant_id.isin(raw.id_participant)
    csv('participant_source_links.csv',links)
    identity_links(people,links)
    inventory=[dict(entity='merged_participants',count=len(people)),dict(entity='selected_participants',count=len(actual)),
        dict(entity='excluded_merged_participants',count=len(people)-len(actual)),dict(entity='merged_recordings',count=len(recordings)),
        dict(entity='original_svc_files',count=len(raw_svc)),dict(entity='participants_with_svc',count=len(raw_ids)),
        dict(entity='raw_metadata_rows',count=len(raw)),dict(entity='raw_metadata_unique_ids',count=raw.id_participant.nunique()),
        dict(entity='raw_metadata_duplicate_id_rows',count=len(raw)-raw.id_participant.nunique()),
        dict(entity='selected_tasks',count=recordings[recordings.included].task.nunique()),dict(entity='selected_recordings',count=int(recordings.included.sum()))]
    csv('data_inventory.csv',pd.DataFrame(inventory))
    by_source=people.groupby(['id_city','included'],dropna=False).size().rename('participants').reset_index()
    csv('selection_by_source.csv',by_source)
    labels=label_table(actual)
    csv('label_consistency.csv',labels);csv('label_disagreements.csv',labels[labels.label_disagreement])
    csv('omission_flag_details.csv',labels[labels.omit_flag])
    label_dict=pd.DataFrame([
        dict(field='group',origin='CSV and merged cognitive metadata',implementation='Actual target for protected models',clinical_definition='UNVERIFIED: grouping/case ascertainment and screening rules absent locally'),
        dict(field='diagnosis',origin='Audit alias of special_care_type_dysgraphia',implementation='Not an independently verified diagnosis variable',clinical_definition='No separate exact diagnosis field assumed; alias explicitly distinguished'),
        dict(field='special_care_type_dysgraphia',origin='CSV and merged metadata',implementation='Specific binary flag, distinct from group',clinical_definition='UNVERIFIED: clinician, diagnostic tool, threshold and assessment timing unknown'),
        dict(field='flag_to_omit_total',origin='Merged metadata; absent from original cognitive CSV',implementation='Recorded but not used by prepare.py eligibility',clinical_definition='Flag components observable in merged; transformation/author intent/official exclusion rule UNVERIFIED')])
    csv('label_dictionary.csv',label_dict)
    association_rows=[];contingencies=[];overlap=[];feasibility=[];control_rows=[];control_metrics=[];missing_rows=[]
    y=actual.label.to_numpy(dtype=int);fold_ids=actual.fold.to_numpy()
    for field in SOURCE_FIELDS:
        values=category(actual[field]);stats,table=association(actual[field],y);association_rows.append(dict(field=field,**stats))
        for key,row in table.iterrows():
            contingency=dict(field=field,source=key,group0=int(row[0]),group1=int(row[1]),n=int(row.sum()))
            contingencies.append(contingency);train0=int((y==0).sum()-row[0]);train1=int((y==1).sum()-row[1])
            feasibility.append(dict(**contingency,train_group0=train0,train_group1=train1,
                heldout_has_two_classes=row[0]>0 and row[1]>0,train_has_two_classes=train0>0 and train1>0,
                algebraically_two_class_loso=bool(row[0]>0 and row[1]>0 and train0>0 and train1>0),
                within_source_minority=int(min(row[0],row[1])),sensitivity_resolution=(1/row[1] if row[1] else np.nan),
                precision_adequacy='Not established; algebraic feasibility is not statistical adequacy'))
        prediction=np.empty(len(y))
        for fold in range(6):
            tr=fold_ids!=fold;te=~tr;train_sources=set(values[tr]);test_sources=set(values[te])
            overlap.append(dict(field=field,fold=fold,train_sources=len(train_sources),test_sources=len(test_sources),
                unseen_test_sources=len(test_sources-train_sources),test_participants=int(te.sum()),
                test_source_seen_in_train=int(values[te].isin(train_sources).sum()),
                fraction_test_source_seen=float(values[te].isin(train_sources).mean()),
                train_source_values='|'.join(sorted(train_sources)),test_source_values='|'.join(sorted(test_sources))))
            for role,mask in [('Train',tr),('Test',te)]:
                for source,part in actual.loc[mask].groupby(category(actual.loc[mask,field])):
                    missing_rows.append(dict(field=field,fold=fold,role=role,source=source,n=len(part),group0=int((part.label==0).sum()),group1=int((part.label==1).sum())))
            prediction[te]=source_majority(values[tr].to_numpy(),y[tr],values[te].to_numpy())
        for i,row in actual.iterrows():control_rows.append(dict(participant_id=row.participant_id,label=int(y[i]),fold=int(fold_ids[i]),control=field,probability=float(prediction[i]),
            mapping_fit_scope='Other five folds only',shrinkage_strength=2,unseen_fallback='Beta(1,1) Train class prior',clinical_classifier=False))
        control_metrics.append(dict(control=field,**metrics(y,prediction)))
    prior=np.empty(len(y))
    for fold in range(6):
        tr=fold_ids!=fold;prior[~tr]=(y[tr].sum()+1)/(tr.sum()+2)
    control_metrics.append(dict(control='Train_class_prior',**metrics(y,prior)))
    for i,row in actual.iterrows():control_rows.append(dict(participant_id=row.participant_id,label=int(y[i]),fold=int(fold_ids[i]),control='Train_class_prior',probability=float(prior[i]),mapping_fit_scope='Other five folds only',shrinkage_strength=2,unseen_fallback='Train prior',clinical_classifier=False))
    csv('source_label_contingency.csv',pd.DataFrame(contingencies));csv('source_association_metrics.csv',pd.DataFrame(association_rows))
    csv('source_fold_overlap.csv',pd.DataFrame(overlap));csv('fold_source_class_distributions.csv',pd.DataFrame(missing_rows))
    csv('metadata_shortcut_control.csv',pd.DataFrame(control_rows));csv('metadata_shortcut_metrics.csv',pd.DataFrame(control_metrics))
    csv('source_validation_feasibility.csv',pd.DataFrame(feasibility))
    miss_rows=[]
    for (city,label),part in actual.groupby(['id_city','label']):
        for field in SOURCE_FIELDS+['age','sex_female','class','language']:
            miss_rows.append(dict(city=city,label=label,field=field,n=len(part),missing=int(missing(part[field]).sum())))
    csv('missingness_by_source_label.csv',pd.DataFrame(miss_rows))
    write_json('source_provenance.json',dict(complete=True,merged_sources=source_hashes,
        raw_cognitive_csv=dict(path=raw_path.relative_to(ROOT).as_posix(),sha256=sha(raw_path)),
        kinematic_source='Merged JSON per-sample arrays -> diagramo.branches.kinematic.extract -> features.kinematic',
        temporal_source='Original SVC -> 21 channels -> float32 per-person cache -> float16 packed RAM input -> fold-Train normalization',
        row_ids_from_participant_files=True,labels_and_sources_joined_exactly=True,cohort_reconstructed=True,
        cohort_changed=False,duplicate_raw_metadata_handling='All values compared to merged; no source row deleted or repaired',
        clinical_label_codebook_available=False,rds_reader_available=False,elapsed_seconds=time.perf_counter()-started))
    write_json('ABC_STATUS.json',dict(complete=True,participants=len(actual),label_disagreements=int(labels.label_disagreement.sum()),omission_flags=int(labels.omit_flag.sum()),
        metadata_diagnostic_only=True,clinical_definition_verified=False,original_cohort_preserved=True))
    print('PHASE A-C COMPLETE',round(time.perf_counter()-started,2),'seconds',flush=True)


if __name__=='__main__':main()
