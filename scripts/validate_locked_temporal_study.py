"""Validate final matched tables, source audit, old successful folds and notebooks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_locked_temporal_data import ROOT,OUT,sha


def main():
    lite=ROOT/'results/raw_temporal_v2/phase4_lite'
    preservation=json.loads((OUT/'successful_fold_preservation.json').read_text())
    for name,digest in preservation['preexisting_completed_results'].items():
        assert sha(lite/name)==digest,('Successful fit overwritten',name)
        original=OUT/'history_9363d15/phase4_lite'/name
        assert sha(original)==digest
    config=json.loads((ROOT/'configs/raw_temporal_v2/phase4_lite.json').read_text())
    assert config['seed']==42 and config['candidates']['stroke_transformer']['fixed_epochs']==28
    assert config['probability_average_weights']==[.5,.5] and config['threshold']==.5
    status=json.loads((OUT/'status.json').read_text());assert status['research_complete'] is False
    frame=pd.read_csv(OUT/'participant_complementarity.csv',float_precision='round_trip')
    assert frame.participant_id.is_unique and frame.fold.nunique()==6 and len(frame)==258
    np.testing.assert_allclose(frame.fixed_average_probability,.5*(frame.probability+frame.kinematic_probability),rtol=0,atol=1e-15)
    expected=np.where((~frame.kinematic_correct)&frame.fusion_correct,'corrected',
        np.where(frame.kinematic_correct&(~frame.fusion_correct),'harmed',np.where(frame.kinematic_correct,'unchanged_correct','unchanged_wrong')))
    np.testing.assert_array_equal(frame.fusion_effect,expected)
    assert status['frozen_stroke_oof_complete'] is True
    audit=json.loads((OUT/'data_audit.json').read_text())
    assert audit['participants']==258 and audit['raw_csv_group_city_site_match']
    sources=pd.read_csv(OUT/'source_audit_participants.csv',keep_default_na=False)
    assert sources.participant_id.is_unique and len(sources)==258
    assert (sources.groupby('alphabetic_prefix').id_city.nunique()==1).all()
    assert (sources.groupby('id_city').alphabetic_prefix.nunique()==1).all()
    city1=sources[sources.id_city=='city1'];other=sources[sources.id_city!='city1']
    assert (city1.label==0).sum()==115 and (city1.label==1).sum()==7
    assert len(other)==136 and other.label.eq(1).all()
    bins=pd.read_csv(OUT/'calibration_bins.csv')
    assert bins.groupby('model').n.sum().eq(258).all()
    assert bins.loc[bins.n==0,'observed_rate'].isna().all()
    book=json.loads((ROOT/'notebooks/09_locked_temporal_study.ipynb').read_text(encoding='utf-8'))
    code=[cell for cell in book['cells'] if cell['cell_type']=='code']
    assert all(cell['execution_count'] is not None for cell in code)
    assert not [out for cell in code for out in cell.get('outputs',[]) if out['output_type']=='error']
    delivery=json.loads((lite/'delivery_checks.json').read_text())
    assert delivery['artifact_checks_passed'] and delivery['requested_lite_folds_complete']
    assert not delivery['original_full_phase4_launched'] and not delivery['auto_push_or_merge']
    result=dict(complete=True,frozen_stroke_oof_complete=True,research_complete=False,
        preexisting_successful_fits_unchanged=len(preservation['preexisting_completed_results']),
        all_258_source_metadata_checked=True,calibration_diagnostics_only=True,
        participant_matched_fixed_fusion_verified=True,executed_notebook_no_errors=True,
        external_validation=False,automatic_followup_training=False,auto_push_or_merge=False,
        outputs_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in [OUT/'LOCKED_RESEARCH_REPORT_VI.md',
            OUT/'matched_oof_metrics.csv',OUT/'paired_bootstrap.csv',OUT/'calibration.png',OUT/'calibration.svg',
            ROOT/'notebooks/09_locked_temporal_study.ipynb']})
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('LOCKED RESEARCH DELIVERY VALIDATED; overall research remains incomplete',flush=True)


if __name__=='__main__':main()
