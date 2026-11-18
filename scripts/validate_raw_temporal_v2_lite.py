"""Final artifact checks and an optional real CUDA interrupted-update test."""
import argparse
from raw_temporal_v2_lite_common import *
from run_raw_temporal_v2_phase4_lite import gradient_update
from session_raw_temporal_v2_lite import checkpoint_io_retry


def cuda_resume_smoke():
    assert DEVICE=='cuda'
    cfg=json.loads(CONFIG.read_text());seed_all(42);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    ids,y,schema,fit,val,test=development();normalizer=normalizer_for(fit)
    model=build_model(cfg['candidates']['duration']['architecture']).to(DEVICE);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(42)
    used=[]
    def step(net,opt,generator):
        assert time.perf_counter()-started<120,'CUDA engineering smoke budget'
        group=generator.permutation(fit)[:2];used.extend(ids[group].tolist());opt.zero_grad(set_to_none=True)
        losses,norm=gradient_update(net,normalizer,ids,y,group,cfg);opt.step();sync()
        return losses,norm,ids[group].tolist()
    step(model,optimizer,rng)
    path=LOCAL/'phase4_lite/engineering_resume.pt'
    atomic_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng)),path)
    expected=step(model,optimizer,rng);parameters={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    restored=build_model(cfg['candidates']['duration']['architecture']).to(DEVICE);restored.train()
    other=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.001)
    state=torch.load(path,map_location=DEVICE,weights_only=False);restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],rng);actual=step(restored,other,rng)
    assert actual==expected and all(torch.equal(value,restored.state_dict()[key].cpu()) for key,value in parameters.items())
    assert not set(used)&set(ids[test])
    result=dict(complete=True,engineering_test_only=True,research_complete=False,
        real_cuda_model='duration',normalizer_fit_ids=ids[fit].tolist(),used_participant_ids=sorted(set(used)),
        interrupted_and_uninterrupted_loss_equal=True,interrupted_and_uninterrupted_weights_bitwise_equal=True,
        optimizer_restored=True,rng_restored=True,elapsed_seconds=time.perf_counter()-started,
        finite_gradients=True,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        checkpoint_sha256=sha(path),checkpoint=path.relative_to(ROOT).as_posix())
    write(LITE/'cuda_resume_validation.json',result);print('REAL CUDA RESUME BITWISE MATCH',result['elapsed_seconds'],flush=True)


def main(cuda=False):
    if cuda:
        with gpu_worker(),awake_during_session(),checkpoint_io_retry():cuda_resume_smoke()
    from report_raw_temporal_v2_lite import main as report
    report()
    cfg=json.loads(CONFIG.read_text());status=json.loads((LITE/'status.json').read_text())
    assert status['research_complete'] is False and status['phase3_complete'] is False
    assert cfg['probability_average_weights']==[.5,.5] and cfg['seed']==42 and not cfg['inner_search']
    for path in (LITE/'sessions').glob('*.json'):
        session=json.loads(path.read_text());assert session['maximum_seconds']<=7200
        if 'elapsed_seconds' in session:assert session['elapsed_seconds']<=session['maximum_seconds'],path
    pilot=json.loads((RECOVERY/'pilot_result.json').read_text());assert not pilot['phase3_complete'] and not pilot['full_ssl_launched']
    assert sha(ROOT/pilot['checkpoint'])==pilot['checkpoint_sha256']
    assert json.loads((RECOVERY/'cuda_smoke.json').read_text())['complete']
    preservation=check_preservation()
    write(LITE/'delivery_checks.json',dict(artifact_checks_passed=True,
        requested_lite_folds_complete=status['lite_validation_complete'],primary_validation_complete=status['primary_validation_complete'],
        research_complete=False,phase3_complete=False,original_full_phase4_launched=False,
        budget_limits_checked=True,preserved=preservation,auto_push_or_merge=False))
    print('LITE DELIVERY CHECKED; research remains incomplete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cuda-resume-smoke',action='store_true');args=parser.parse_args();main(args.cuda_resume_smoke)
