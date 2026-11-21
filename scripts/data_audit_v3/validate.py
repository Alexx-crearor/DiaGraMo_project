"""Artifact, provenance and non-mutation validation; no clinical estimators fitted."""
from __future__ import annotations
import datetime
import importlib.metadata
from typing import Any
from common import *
from phase_abc import source_majority,category


def legacy_expectations() -> dict[str,str]:
    legacy=json.loads((ROOT/'results/raw_temporal/protected_manifest.json').read_text())
    saved=json.loads((ROOT/'results/raw_temporal/reproducibility_manifest.json').read_text())
    legacy.update(saved['source_sha256']);legacy.update(saved['artifact_sha256'])
    return legacy


def supplemental_legacy() -> None:
    before=read('PROTECTED_BEFORE.csv').set_index('path');extra=[]
    for name,expected in legacy_expectations().items():
        if name in before.index:continue
        actual=sha(ROOT/name)
        if actual!=expected:raise AssertionError(('Historical checkpoint/artifact mismatch',name))
        extra.append(dict(path=name,bytes=(ROOT/name).stat().st_size,sha256=actual,expected_sha256=expected,
            provenance='Historical immutable manifest expectation; first rehashed in validation, not retroactively added to initial snapshot'))
    csv('LEGACY_PROTECTED_ADDITIONAL.csv',pd.DataFrame(extra,columns=['path','bytes','sha256','expected_sha256','provenance']))


def manifest() -> None:
    supplemental_legacy()
    items=[]
    for folder in [OUT,ROOT/'scripts/data_audit_v3',ROOT/'tests/data_audit_v3']:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name in ['ARTIFACT_MANIFEST.json','VALIDATION_CHECKS.json']:continue
            if path.is_relative_to(ROOT/'scripts/data_audit_v3') or path.is_relative_to(ROOT/'tests/data_audit_v3'):provenance='New read-only audit source/test code'
            elif path.name.startswith(('sensor','preprocessing','acquisition','svc_merged','normalization')):provenance='Original local SVC + protected caches; participant/recording summaries'
            elif path.name.startswith(('matched','calibration','error','fusion','label_sensitivity','paired_model')):provenance='Existing protected OOF only, plus source metadata joins'
            else:provenance='Local CSV/merged metadata, source inventory and audit synthesis; see phase scripts'
            items.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path),provenance=provenance))
    write_json('ARTIFACT_MANIFEST.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=42,
        reference_commit='35ad4208ae5a8180942a00d8e89a06df3d7ac813',artifacts=items,
        excluded_from_self_hash=['ARTIFACT_MANIFEST.json','VALIDATION_CHECKS.json'],raw_source_copies_in_outputs=False,
        dependencies={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','pytest']}))


def main() -> None:
    expected=['data_inventory.csv','cohort_selection_flow.csv','metadata_dictionary.csv','source_provenance.json',
        'label_dictionary.csv','label_consistency.csv','label_disagreements.csv','label_sensitivity.csv','source_label_contingency.csv',
        'source_association_metrics.csv','source_fold_overlap.csv','metadata_shortcut_control.csv','source_validation_feasibility.csv',
        'sensor_quality_by_participant.csv','sensor_quality_by_source.csv','sensor_quality_by_task.csv','preprocessing_fidelity.csv',
        'acquisition_signature_analysis.csv','matched_model_metrics.csv','source_stratified_model_metrics.csv','error_overlap_by_source.csv',
        'fusion_corrected_harmed.csv','calibration_diagnostics.csv','COMPREHENSIVE_DATA_AUDIT_VI.md','EXECUTIVE_SUMMARY_VI.md','REPRODUCE.md','BLOCKERS.md']
    for phase in 'ABCDEF':expected.append(f'PHASE_{phase}_REPORT_VI.md')
    for name in expected:
        if not (OUT/name).is_file():raise AssertionError(('Missing deliverable',name))
    for phase in ['ABC','D','EF']:assert json.loads((OUT/f'{phase}_STATUS.json').read_text())['complete']
    assignment=folds();cohort=read('audit_cohort.csv');assert cohort.participant_id.is_unique and len(cohort)==len(assignment)
    np.testing.assert_array_equal(cohort.participant_id,assignment.participant_id);np.testing.assert_array_equal(cohort.label,assignment.label);np.testing.assert_array_equal(cohort.fold,assignment.fold)
    all_people=read('participants_inventory.csv');assert all_people.participant_id.is_unique
    assert set(all_people.loc[all_people.included,'participant_id'])==set(cohort.participant_id)
    links=read('participant_source_links.csv');assert links.loc[links.included,'metadata_id_matches_filename'].all()
    assert links.loc[links.included,'has_original_svc'].all() and links.loc[links.included,'has_raw_cognitive_id'].all()
    full=read('merged_recording_inventory.csv');selected=full[full.included]
    assert selected.groupby('participant_id').size().eq(len(TASKS)).all()
    assert selected[['participant_id','task']].drop_duplicates().shape[0]==len(selected)
    assert set(selected.task)==set(TASKS)
    joins=read('metadata_join_checks.csv');unverifiable=joins.verification.str.startswith('UNVERIFIABLE')
    assert joins.loc[unverifiable,'matches_merged'].eq('').all(),'Unverified source fields cannot pass as verified'
    verified=joins[~unverifiable&joins.participant_id.isin(cohort.participant_id)]
    assert verified.matches_merged.astype(str).eq('True').all()
    # Recompute control probabilities from Train arrays, not a full-cohort majority map.
    control=read('metadata_shortcut_control.csv');y=cohort.label.to_numpy();foldids=cohort.fold.to_numpy()
    for field in ['id_city','id_school_ppp','site_type_ppp','date_of_administration']:
        values=category(cohort[field]).to_numpy();observed=control[control.control==field].set_index('participant_id').loc[cohort.participant_id]
        for fold in range(6):
            tr=foldids!=fold;te=~tr;expected_p=source_majority(values[tr],y[tr],values[te])
            np.testing.assert_allclose(observed.probability.to_numpy()[te],expected_p,rtol=0,atol=1e-15)
    fidelity=read('preprocessing_fidelity.csv');assert len(fidelity)==len(cohort)*len(TASKS)
    for field in ['float32_cache_exact','index_cache_exact','packed_float16_exact','independent_masks_exact','sample_coverage_exact','source_hash_matches_manifest']:assert fidelity[field].all()
    assert read('normalization_statistics_replay.csv').statistics_match.all()
    oof=read('matched_oof_replay.csv');assert len(oof)==len(cohort) and oof.participant_id.is_unique
    np.testing.assert_array_equal(oof.fold,assignment.fold);np.testing.assert_array_equal(oof.label,assignment.label)
    np.testing.assert_allclose(oof.pF,.5*(oof.pK+oof.pT),rtol=0,atol=1e-15)
    strata=read('source_stratified_model_metrics.csv');single=(strata.group0==0)|(strata.group1==0)
    assert strata.loc[single,'auc'].eq('').all() and strata.loc[single,'balanced_accuracy'].eq('').all()
    # Check older protection manifests against the new independently captured snapshot.
    before=read('PROTECTED_BEFORE.csv').set_index('path');legacy=legacy_expectations()
    additional=read('LEGACY_PROTECTED_ADDITIONAL.csv').set_index('path')
    for name,digest in legacy.items():
        actual=before.loc[name,'sha256'] if name in before.index else additional.loc[name,'sha256']
        assert actual==digest,('Legacy protected mismatch',name)
    tests=json.loads((OUT/'TEST_RESULTS.json').read_text());assert tests['complete'] and tests['returncode']==0
    protected=verify_protection()
    artifact=json.loads((OUT/'ARTIFACT_MANIFEST.json').read_text())
    for item in artifact['artifacts']:assert sha(ROOT/item['path'])==item['sha256'],('Audit artifact changed',item['path'])
    write_json('VALIDATION_CHECKS.json',dict(audit_computationally_complete=True,completed_phases=['A','B','C','D','E','F'],
        cohort_reconstruction_exact=True,label_and_metadata_joins_checked=True,
        unverifiable_raw_csv_fields=sorted(joins.loc[unverifiable,'field'].unique().tolist()),
        source_only_control_train_scope_recomputed=True,source_folds_verified=True,
        raw_and_cache_fidelity_verified=True,all_oof_matched=True,one_class_auc_not_fabricated=True,
        protected_artifacts=protected,legacy_protected_references=len(legacy),tests=tests,
        supplemental_historical_hash_references_checked=len(additional),
        artifact_hashes_checked=len(artifact['artifacts']),clinical_label_definition_verified=False,
        device_causation_verified=False,source_independent_validation_established=False,
        overall_research_complete=False,raw_sensitive_data_committed=False,auto_push_or_merge=False,
        blockers_documented=True,interpretation='Audit completed with documented epistemic limits; clinical/source-invariant validity not certified'))
    print('COMPREHENSIVE AUDIT VALIDATED; CLINICAL VALIDITY NOT CERTIFIED',flush=True)


if __name__=='__main__':manifest();main()
