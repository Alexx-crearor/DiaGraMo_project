"""One-off numerical audit of the latency warmup correction, not model selection."""
import argparse
from raw_temporal_common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folds',nargs='+',type=int,default=[0,1]);args=parser.parse_args()
    for fold in args.folds:
        assert fold in (0,1)
        old=ROOT/'data/derived/raw_temporal_xai_pre_latency_warmup'/f'fold_{fold}'
        new=OUT/'xai'/f'fold_{fold}';assert json.loads((new/'complete.json').read_text())['complete']
        comparisons=[]
        for name in ('task_occlusion','segment_occlusion','stroke_occlusion','temporal_occlusion','perturbation_oof','embeddings','attention_supplementary'):
            a=pd.read_csv(old/(name+'.csv'),float_precision='round_trip');b=pd.read_csv(new/(name+'.csv'),float_precision='round_trip')
            assert a.shape==b.shape and a.columns.tolist()==b.columns.tolist();maximum=0.
            for column in a:
                if pd.api.types.is_numeric_dtype(a[column]):
                    np.testing.assert_allclose(a[column],b[column],atol=3e-6,rtol=0)
                    maximum=max(maximum,float(np.max(np.abs(a[column].to_numpy()-b[column].to_numpy()))))
                else:np.testing.assert_array_equal(a[column],b[column])
            comparisons.append(dict(file=name+'.csv',rows=len(a),maximum_absolute_difference=maximum,
                original_sha256=sha(old/(name+'.csv')),warmup_sha256=sha(new/(name+'.csv'))))
        representation_maximum=0.
        for path in old.glob('*_representations.npz'):
            with np.load(path)as a,np.load(new/path.name)as b:
                for name in ('ids','tasks'):np.testing.assert_array_equal(a[name],b[name])
                for name in ('z_task','z_person'):
                    x=a[name];y=b[name];np.testing.assert_allclose(x,y,atol=3e-6,rtol=0)
                    representation_maximum=max(representation_maximum,float(np.max(np.abs(x-y))))
        write(OUT/f'xai_warmup_equivalence_fold_{fold}.json',dict(checks_passed=True,fold=fold,
            scope='Same fixed checkpoints and CPU explanations; latency measurements intentionally differ',
            csv_comparisons=comparisons,representation_maximum_absolute_difference=representation_maximum,
            checker_sha256=sha(Path(__file__)),tolerance=3e-6))
        print('XAI WARMUP OUTPUT EQUIVALENCE PASSED',fold,flush=True)

if __name__=='__main__':main()
