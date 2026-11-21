"""Descriptive source-confounding audit. Never fit a predictor using IDs/metadata."""
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/raw_temporal_v2/locked_study'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def association(frame, column):
    table=pd.crosstab(frame[column].fillna('<missing>'),frame.label).reindex(columns=[0,1],fill_value=0)
    n=int(table.to_numpy().sum())
    chi2=float(chi2_contingency(table,correction=False)[0]) if len(table)>1 else 0.
    return dict(field=column,participants=n,categories=len(table),
        mixed_label_categories=int((table.gt(0).sum(axis=1)==2).sum()),
        single_label_categories=int((table.gt(0).sum(axis=1)==1).sum()),
        descriptive_label_purity=float(table.max(axis=1).sum()/n),
        cramers_v=float(np.sqrt(chi2/n)) if len(table)>1 else 0.),table


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')
    assert len(folds)==258 and folds.participant_id.is_unique
    fields=['id_city','id_school_ppp','site_type_ppp','date_of_administration']
    rows=[];sources=[];allkeys=set();decoder=json.JSONDecoder()
    for i,row in enumerate(folds.itertuples(index=False)):
        path=ROOT/'data/clean_data'/f'{row.participant_id}_merged.json'
        text=path.read_text(encoding='utf-8')
        match=re.search(r'"[^"\\]+_cognitive_test_data"\s*:\s*\[\s*',text)
        assert match,path
        metadata,_=decoder.raw_decode(text,match.end());allkeys.update(metadata)
        assert metadata['id_participant']==row.participant_id and int(metadata['group'])==row.label
        prefix=re.match(r'[A-Za-z]+',row.participant_id).group()
        rows.append(dict(participant_id=row.participant_id,label=row.label,fold=row.fold,
            alphabetic_prefix=prefix,prefix_first5=row.participant_id[:5],
            **{field:metadata.get(field) for field in fields}))
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
        if i%50==0:print('SOURCE METADATA AUDITED',i+1,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'source_audit_participants.csv',index=False)
    summaries=[];tables=[]
    for column in ['alphabetic_prefix','prefix_first5']+fields:
        summary,table=association(frame,column);summaries.append(summary)
        table=table.rename(columns={0:'group0',1:'group1'}).reset_index().rename(columns={column:'category'})
        table.insert(0,'field',column);tables.append(table)
    pd.DataFrame(summaries).to_csv(OUT/'source_associations.csv',index=False)
    pd.concat(tables,ignore_index=True).to_csv(OUT/'source_label_contingencies.csv',index=False)
    mapping=[]
    for prefix,group in frame.groupby('alphabetic_prefix'):
        mapping.append(dict(prefix=prefix,n=len(group),cities='|'.join(sorted(group.id_city.dropna().unique())),
            city_count=group.id_city.nunique(),sites=group.id_school_ppp.nunique(),
            group0=int((group.label==0).sum()),group1=int((group.label==1).sum())))
    pd.DataFrame(mapping).to_csv(OUT/'prefix_city_mapping.csv',index=False)
    coverage=[]
    for fold in range(6):
        tr=frame[frame.fold!=fold];te=frame[frame.fold==fold]
        for column in ['alphabetic_prefix','id_city','id_school_ppp','date_of_administration']:
            coverage.append(dict(fold=fold,field=column,test_participants=len(te),
                test_with_category_already_in_train=int(te[column].isin(tr[column].dropna()).sum()),
                test_missing=int(te[column].isna().sum())))
    pd.DataFrame(coverage).to_csv(OUT/'source_fold_overlap.csv',index=False)
    rawcsv=ROOT/'data/raw_data/cognitive_test_data/cognitiveTestCSV2data.csv'
    rawhead=pd.read_csv(rawcsv,sep=None,engine='python',nrows=0)
    rawmeta=pd.read_csv(rawcsv,sep=None,engine='python',usecols=['id_participant','group','id_city','id_schoolPPP'])
    duplicated=rawmeta[rawmeta.id_participant.duplicated(False)]
    assert (duplicated.groupby('id_participant').nunique(dropna=False)<=1).all().all(), 'Conflicting source metadata'
    duplicate_rows=len(rawmeta)-rawmeta.id_participant.nunique()
    rawmeta=rawmeta.drop_duplicates('id_participant')
    rawids=rawmeta.id_participant
    linked=rawmeta.set_index('id_participant').reindex(frame.participant_id)
    assert linked['group'].notna().all()
    assert np.array_equal(linked['group'].to_numpy(),frame.label.to_numpy())
    assert np.array_equal(linked.id_city.to_numpy(),frame.id_city.to_numpy())
    assert np.array_equal(linked.id_schoolPPP.to_numpy(),frame.id_school_ppp.to_numpy())
    result=dict(complete=True,participants=258,labels_verified=True,
        metadata_source='Merged participant-linked cognitive metadata, exact participant-ID join',
        raw_metadata_csv_sha256=sha(rawcsv),raw_csv_exact_id_matches=int(frame.participant_id.isin(rawids).sum()),
        raw_csv_group_city_site_match=True,
        raw_csv_duplicate_id_rows=duplicate_rows,
        raw_csv_duplicate_id_fields_consistent=['group','id_city','id_schoolPPP'],
        merged_metadata_fields=sorted(allkeys),metadata_sources=sources,
        hardware_batch_recruitment_fields_found=[k for k in sorted(allkeys) if re.search('hardware|tablet|device|batch|recruit',k,re.I)],
        field_missing_counts={k:int(frame[k].isna().sum()) for k in fields},
        descriptive_only=True,metadata_used_for_prediction=False,
        id_prefix_used_for_prediction=False,source_disjoint_validation=False,
        raw_metadata_relevant_columns=[k for k in rawhead if re.search('city|school|site|date|hardware|tablet|batch|recruit',k,re.I)],
        elapsed_seconds=time.perf_counter()-started)
    (OUT/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DATA VALIDITY AUDIT COMPLETE',result['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
