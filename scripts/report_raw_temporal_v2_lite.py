"""Artifact-only recovery and paired OOF reporting, including incomplete folds."""
from raw_temporal_v2_lite_common import *
from sklearn.metrics import confusion_matrix


def markdown(frame):
    if frame.empty:return 'Không có hàng dữ liệu trong bảng này.'
    def cell(value):
        if isinstance(value,(float,np.floating)):return '' if not np.isfinite(value) else f'{value:.6f}'
        return str(value).replace('|','/').replace('\n',' ')
    header='| '+' | '.join(map(str,frame.columns))+' |'
    separator='| '+' | '.join(['---']*len(frame.columns))+' |'
    return '\n'.join([header,separator,*('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))])


def recovery_report():
    smoke_path=RECOVERY/'cuda_smoke.json';pilot_path=RECOVERY/'pilot_result.json'
    if not smoke_path.exists():return
    smoke=json.loads(smoke_path.read_text());pilot=json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    text=f'''# Phase 3 recovery — kiểm tra CUDA và ngân sách

**Phase 3 chưa hoàn tất.** Đã xác minh bản sửa ở commit `05b3e28` trên dữ liệu thật, strict deterministic CUDA. Lỗi trước đây nằm ở backward của nội suy tuyến tính; bản sửa dùng toán tử tuyến tính tương đương bằng GEMM, giữ nguyên vị trí các tâm convolution. Không đổi kiến trúc hoặc mục tiêu SSL.

{markdown(pd.DataFrame(smoke['records']))}

Normalizer chỉ fit 143 development-Train participants; dữ liệu smoke lấy từ development Train, không dùng 43 outer Test0. Hai lần forward/backward với cùng weights/input tạo kết quả bit-for-bit bằng nhau. Đây là kiểm tra vận hành, không đo chất lượng embedding hay chẩn đoán.
'''
    if pilot:
        keys=['epochs_completed','optimizer_updates','eligible_windows','training_seconds','forward_only_seconds',
              'windows_per_second','peak_gpu_allocated_mib','peak_gpu_reserved_mib',
              'estimated_full_train_epoch_seconds','estimated_validation_forward_seconds',
              'estimated_two_seed_pretraining_upper_seconds','estimated_full_phase3_seconds',
              'configured_full_phase3_budget_seconds','full_suite_estimate_within_budget']
        text+='\n## Epoch pilot và ước lượng\n\n'+markdown(pd.DataFrame([{'quantity':k,'value':pilot[k]} for k in keys]))
        text+=f'''

Epoch nhỏ gồm 8 người được lấy theo quantile số window trong development Train; dùng toàn bộ window pen-down hợp lệ của mỗi người, một optimizer update với gradient cân bằng theo participant. Normalizer dùng pool 143 Train. Window pen-up không có reconstruction objective vẫn nằm trong dữ liệu classification; không loại khỏi pipeline classifier.

Đây là một epoch trên tập con, không phải một epoch đầy đủ trên 143 người. Ước lượng theo tỷ lệ window: train epoch khoảng {pilot['estimated_full_train_epoch_seconds']:.2f}s; reconstruction validation forward khoảng {pilot['estimated_validation_forward_seconds']:.2f}s. Hai pretraining seed × tối đa 8 epochs, cộng safety factor 1,5, khoảng {pilot['estimated_two_seed_pretraining_upper_seconds']/60:.2f} phút. Validation được benchmark forward-only nên chưa bao gồm toàn bộ baseline/masking/ghi file; không coi đây là thời gian cam kết.

Toàn bộ Phase 3 còn gồm 12 neural downstream fits và 38 CPU readout fits. Dùng thời gian downstream duration-aware đã đo để proxy, tổng khoảng {pilot['estimated_full_phase3_seconds']/60:.2f} phút; chưa cộng đủ CPU/extraction và uncertainty. Ngân sách riêng đặt cho full Phase 3 là {pilot['configured_full_phase3_budget_seconds']/60:.0f} phút để dành phần còn lại của phiên cho validation nhẹ. Không khởi chạy full SSL. Trạng thái giữ **pilot-only/incomplete**, dù pretraining riêng lẻ có thể khả thi.

Checkpoint pilot: `{pilot['checkpoint']}` (cache cục bộ), SHA `{pilot['checkpoint_sha256']}`. Không dùng checkpoint này để tạo embedding toàn cohort hoặc đưa vào CV. Chuẩn hóa/encoder đã thấy pool 143 người nên không được tái sử dụng tùy tiện cho các fold khác.
'''
    text+='''

Khuyến nghị: ưu tiên OOF của mô hình scratch đã có tín hiệu development trước. Chỉ lên lịch full SSL sau khi thấy nhu cầu bổ sung thông tin và khóa ngân sách cho cả matched scratch/random/frozen/fine-tune controls; reconstruction loss đơn lẻ không đủ để khẳng định embedding hữu ích.
'''
    (OUT/'PHASE3_RECOVERY_VI.md').write_text(text,encoding='utf-8',newline='\n')


def main():
    cfg=json.loads(CONFIG.read_text());ids,y,schema,assignments,splits=cohort();sig=model_signature()
    kin=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv',float_precision='round_trip').set_index('participant_id').loc[ids]
    np.testing.assert_array_equal(kin.label,y);np.testing.assert_array_equal(kin.fold,assignments.fold)
    frames=[];costs=[];validation=[]
    for path in sorted((LITE/'fits').glob('*/result.json')):
        result=json.loads(path.read_text());name=result['model'];fold=result['fold'];train,test=splits[fold]
        assert result['complete'] and result['signature']==sig['signature']
        assert result['fixed_epochs']==cfg['candidates'][name]['fixed_epochs'] and result['seed']==42
        assert set(result['fit_ids'])==set(ids[train]) and set(result['normalizer_fit_ids'])==set(ids[train])
        assert set(result['evaluation_ids'])==set(ids[test]) and result['validation_ids']==[]
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        state=torch.load(ROOT/result['checkpoint'],map_location='cpu',weights_only=False)
        assert set(state['fit_ids'])==set(ids[train]) and set(state['evaluation_ids'])==set(ids[test])
        assert state['validation_ids']==[] and state['fixed_epochs']==result['fixed_epochs']
        expected_normalizer=normalizer_for(train)
        assert set(state['normalizer'].fit_ids)==set(ids[train])
        np.testing.assert_allclose(state['normalizer'].mean,expected_normalizer.mean,atol=1e-12,rtol=0)
        np.testing.assert_allclose(state['normalizer'].scale,expected_normalizer.scale,atol=1e-12,rtol=0)
        assert all(torch.isfinite(value).all() for value in state['state'].values())
        resume=torch.load((ROOT/result['checkpoint']).with_suffix('.resume.pt'),map_location='cpu',weights_only=False)
        assert resume['epoch']==result['fixed_epochs'] and resume['cursor']==0 and resume['order'] is None
        assert resume['optimizer_steps']==result['fixed_epochs']*int(np.ceil(len(train)/cfg['training']['effective_batch']))
        assert all(torch.equal(value,resume['model'][key]) for key,value in state['state'].items())
        assert result['peak_gpu_allocated_mib']<=cfg['compute']['gpu_tensor_limit_mib']
        frame=pd.read_csv(path.parent/'oof_predictions.csv',float_precision='round_trip').set_index('participant_id').loc[ids[test]].reset_index()
        np.testing.assert_array_equal(frame.label,y[test]);assert (frame.fold==fold).all() and (frame.model==name).all()
        np.testing.assert_allclose(frame.probability,1/(1+np.exp(-frame.logit)),atol=1e-12,rtol=0)
        assert (frame.checkpoint_sha256==result['checkpoint_sha256']).all()
        for metric,value in metrics(frame.label,frame.probability).items():assert abs(value-result['metrics'][metric])<1e-10
        frames.append(frame);costs.append(dict(model=name,fold=fold,training_seconds=result['training_seconds'],
            evaluation_and_replay_seconds=result['evaluation_and_replay_seconds'],peak_gpu_allocated_mib=result['peak_gpu_allocated_mib'],
            peak_gpu_reserved_mib=result['peak_gpu_reserved_mib'],parameters=result['parameters'],epochs=result['fixed_epochs']))
        validation.append(dict(model=name,fold=fold,participants=len(frame),checkpoint_replay_error=result['checkpoint_replay_max_logit_error'],
            fit_test_disjoint=True,normalizer_train_only=True,outer_test_not_used_for_selection=True))
    recovery_report()
    if not frames:
        write(LITE/'status.json',dict(lite_validation_complete=False,research_complete=False,status='no_completed_folds',
            primary_folds_completed=0,secondary_folds_completed=0,phase3_complete=False,full_phase4_launched=False))
        (OUT/'PHASE4_LITE_REPORT_VI.md').write_text('# Phase 4 lite\n\nChưa có fold hoàn tất. Cấu hình đã khóa; đang chờ phiên chạy trong ngân sách.\n',encoding='utf-8')
        return
    oof=pd.concat(frames,ignore_index=True);assert not oof.duplicated(['model','participant_id']).any()
    oof['kinematic_probability']=kin.loc[oof.participant_id,'probability'].to_numpy()
    oof['fixed_average_probability']=fixed_average(oof.kinematic_probability,oof.probability)
    oof['error_category']=paired_error_categories(oof.label.to_numpy(),oof.kinematic_probability.to_numpy(),oof.probability.to_numpy())
    oof['raw_minus_kinematic_probability']=oof.probability-oof.kinematic_probability
    oof.to_csv(LITE/'oof_predictions.csv',index=False)
    summary=[];fold_rows=[];overlap=[];deltas=[]
    for name,group in oof.groupby('model',sort=False):
        complete=group.fold.nunique()==6 and len(group)==258
        for method,column in [('raw','probability'),('kinematic','kinematic_probability'),('fixed_average_50_50','fixed_average_probability')]:
            values=metrics(group.label,group[column]);summary.append(dict(candidate=name,method=method,
                participants=len(group),completed_folds=group.fold.nunique(),all_six_folds_complete=complete,**values))
            for fold,a in group.groupby('fold'):fold_rows.append(dict(candidate=name,method=method,fold=fold,participants=len(a),**metrics(a.label,a[column])))
        counts=group.error_category.value_counts().to_dict();k=group.kinematic_probability.to_numpy();r=group.probability.to_numpy();truth=group.label.to_numpy()
        overlap.append(dict(candidate=name,participants=len(group),**{c:int(counts.get(c,0)) for c in ['both_correct','kin_only','raw_only','both_wrong']},
            probability_correlation=float(np.corrcoef(k,r)[0,1]) if np.std(k)>0 and np.std(r)>0 else None))
        # Conditional paired bootstrap, same participant resamples for each method.
        rng=np.random.default_rng(42);samples={method:{metric:[] for metric in ['f1_macro','auc','balanced_accuracy','specificity']} for method in ['raw','fixed_average_50_50']}
        for repeat in range(500):
            at=rng.integers(0,len(group),len(group))
            if len(np.unique(truth[at]))<2:continue
            reference=metrics(truth[at],k[at])
            for method,p in [('raw',r),('fixed_average_50_50',group.fixed_average_probability.to_numpy())]:
                value=metrics(truth[at],p[at])
                for metric in samples[method]:samples[method][metric].append(value[metric]-reference[metric])
        for method,columns in samples.items():
            p=r if method=='raw' else group.fixed_average_probability.to_numpy()
            for metric,values in columns.items():
                lo,hi=np.quantile(values,[.025,.975]);deltas.append(dict(candidate=name,method=method,reference='kinematic_same_participants',metric=metric,
                    participants=len(group),delta=metrics(truth,p)[metric]-metrics(truth,k)[metric],lower95=lo,upper95=hi,bootstrap_replicates=len(values)))
    summary=pd.DataFrame(summary);folds=pd.DataFrame(fold_rows);costs=pd.DataFrame(costs);delta=pd.DataFrame(deltas);overlap=pd.DataFrame(overlap)
    fold_summary=folds.groupby(['candidate','method'])[['accuracy','f1_macro','auc','balanced_accuracy','specificity']].agg(['mean','std'])
    fold_summary.columns=['fold_'+a+'_'+b for a,b in fold_summary.columns];summary=summary.merge(fold_summary.reset_index(),on=['candidate','method'])
    for name,frame in [('paired_comparison',summary),('fold_metrics',folds),('runtime_memory',costs),('error_overlap',overlap),('paired_deltas',delta),('validation_checks',pd.DataFrame(validation))]:
        frame.to_csv(LITE/(name+'.csv'),index=False)
    pending=[]
    for path in sorted((LITE/'fits').glob('*/status.json')):
        if (path.parent/'result.json').exists():continue
        state_path=LOCAL/'phase4_lite'/(path.parent.name+'.resume.pt')
        if not state_path.exists():continue
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        pending.append(dict(fit=path.parent.name,epochs_completed=state['epoch'],fixed_epochs=state['fixed_epochs'],
            next_group_cursor=state['cursor'],optimizer_steps=state['optimizer_steps'],training_seconds=state['training_seconds'],
            peak_gpu_allocated_mib=state['peak_gpu_allocated_mib'],peak_gpu_reserved_mib=state['peak_gpu_reserved_mib'],
            oof_available=False,resumable_checkpoint=state_path.relative_to(ROOT).as_posix()))
    pending_table=pd.DataFrame(pending,columns=['fit','epochs_completed','fixed_epochs','next_group_cursor','optimizer_steps','training_seconds',
        'peak_gpu_allocated_mib','peak_gpu_reserved_mib','oof_available','resumable_checkpoint'])
    pending_table.to_csv(LITE/'incomplete_fit_runtime.csv',index=False)
    empirical=[]
    for name,group in costs.groupby('model'):
        per_fold=float((group.training_seconds+group.evaluation_and_replay_seconds).mean())
        empirical.append(dict(candidate=name,measured_completed_folds=len(group),actual_completed_training_and_replay_seconds=float((group.training_seconds+group.evaluation_and_replay_seconds).sum()),
            mean_actual_fold_seconds=per_fold,estimated_remaining_full_folds_seconds=(6-len(group))*per_fold,
            note='Runtime extrapolation only, does not change architecture/epochs; unfinished-fold work not subtracted'))
    empirical_table=pd.DataFrame(empirical);empirical_table.to_csv(LITE/'empirical_runtime_forecast.csv',index=False)
    completed={name:len(list((LITE/'fits').glob(name+'_fold*/result.json'))) for name in cfg['candidates']}
    all_complete=all(count==6 for count in completed.values())
    status=dict(lite_validation_complete=all_complete,primary_validation_complete=completed['duration']==6,
        research_complete=False,status='lite_complete' if all_complete else 'lite_incomplete_resumable',
        primary_folds_completed=completed['duration'],secondary_folds_completed=completed['stroke_transformer'],
        phase3_complete=False,full_phase4_launched=False,exploratory=True,checkpoint_checks_passed=True)
    write(LITE/'status.json',status)
    preservation=check_preservation();write(LITE/'preservation_validation.json',preservation)
    (LITE/'figures').mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
    fig,ax=plt.subplots(figsize=(10,5))
    labels=[f'{r.candidate}: {r.method} (n={r.participants})' for r in summary.itertuples()]
    ax.scatter(summary.f1_macro,np.arange(len(summary)),color='#24506B');ax.set_yticks(range(len(summary)),labels)
    ax.set_xlim(0,1);ax.set_xlabel('Held-out macro F1 on matched participants');ax.invert_yaxis();fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(LITE/'figures'/('paired_f1.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    forecast=json.loads((LITE/'runtime_forecast.json').read_text()) if (LITE/'runtime_forecast.json').exists() else {}
    forecast_table=pd.DataFrame([dict(candidate=k,**{q:v for q,v in a.items() if q in ['training_seconds','estimated_outer_train_epoch_seconds','estimated_fold_seconds_with_safety','estimated_six_folds_seconds_with_safety']}) for k,a in forecast.get('candidate_forecasts',{}).items()])
    primary_raw=delta[(delta.candidate=='duration')&(delta.method=='raw')&(delta.metric=='f1_macro')].iloc[0]
    primary_average=delta[(delta.candidate=='duration')&(delta.method=='fixed_average_50_50')&(delta.metric=='f1_macro')].iloc[0]
    recommendation=f"Duration-aware trên {int(primary_raw.participants)} người hiện có ΔF1 Raw−Kinematic={primary_raw.delta:+.6f}, average−Kinematic={primary_average.delta:+.6f}. "
    if completed['duration']<6:
        recommendation+='Primary còn thiếu folds: chưa có cơ sở kết luận về full OOF; ưu tiên hoàn tất protocol cố định bằng các phiên ngắn.'
    elif primary_average.lower95>0:
        recommendation+='Average có CI F1 dương trong phân tích thăm dò này; có lý do kiểm chứng lại bằng một shortlist nhỏ và Train-only selection khi có ngân sách, trước khi mở rộng kiến trúc hoặc SSL.'
    else:
        recommendation+='Average chưa có CI F1 hoàn toàn dương so với Kinematic. Chưa có bằng chứng đủ mạnh để ưu tiên full nested fusion/SSL; giữ baseline và hoàn tất đối chứng cố định còn thiếu trước.'
    secondary=delta[(delta.candidate=='stroke_transformer')&(delta.method=='fixed_average_50_50')&(delta.metric=='f1_macro')]
    if len(secondary):
        row=secondary.iloc[0]
        recommendation+=f" Stroke Transformer average trên {int(row.participants)} người hiện có ΔF1={row.delta:+.6f}, CI95% [{row.lower95:+.6f},{row.upper95:+.6f}]."
        if completed['stroke_transformer']<6:
            recommendation+=f" Đây mới là partial OOF; nếu có tín hiệu thuận lợi thì bước hợp lý là hoàn tất {6-completed['stroke_transformer']} folds còn thiếu theo protocol đã định trước trong các phiên ngắn, không chọn kiến trúc hoặc trọng số mới từ các outer scores này."
    session_rows=[]
    for path in sorted((LITE/'sessions').glob('*.json')):
        record=json.loads(path.read_text());session_rows.append(dict(session=path.stem,status=record['status'],
            maximum_seconds=record['maximum_seconds'],elapsed_seconds=record.get('elapsed_seconds'),
            reporting_seconds=record.get('reporting_seconds')))
    session_table=pd.DataFrame(session_rows)
    text=f'''# Phase 4 lite — đánh giá cấu hình cố định trong ngân sách

**Trạng thái:** `{status['status']}`. Primary duration-aware: **{completed['duration']}/6 folds**; Stroke Transformer: **{completed['stroke_transformer']}/6 folds**. Phase 3 full SSL và nghiên cứu đầy đủ **chưa hoàn tất**. Protocol Phase 4 đầy đủ được giữ nguyên và không khởi chạy.

## Protocol khóa trước outer evaluation

258 participants, nhãn `group`, original participant-independent Stratified 6-Fold seed42. Một initialization seed42, không inner architecture search. Primary là duration-aware, chọn bằng kết quả development Phase1/2 đã có; secondary là Stroke Transformer. Mỗi mô hình fit mới trên outer Train215, normalizer chỉ fit Train. Primary **22 epochs**, secondary **28 epochs**, lấy rounded median BCE-best development epochs ở hai seed. AdamW LR0.001, weight decay0.001, dropout0.1, clip1, effective batch8/micro2. Không dùng outer Test để early-stop, chọn mô hình, epoch, threshold hoặc fusion weight. Không kéo dài fit dựa vào outer score.

Cả hai mô hình khởi tạo ngẫu nhiên với seed42 và học supervised từ đầu; không dùng checkpoint SSL pilot trong OOF.

Đây là **exploratory internal CV**: cùng cohort đã được xem trong lịch sử và development toàn cục ảnh hưởng cấu hình; không mô tả đây là Nested CV hoặc xác nhận độc lập.

## So sánh OOF trên đúng cùng participants

{markdown(summary[['candidate','method','participants','completed_folds','all_six_folds_complete','accuracy','f1_macro','auc','balanced_accuracy','specificity']])}

![Matched held-out macro F1](phase4_lite/figures/paired_f1.png)

Mỗi dòng Kinematic được giới hạn đúng IDs/folds có Raw hoàn tất của candidate đó. Chỉ khi 6/6 folds và258 IDs mới gọi là full-cohort OOF. Fold còn thiếu không được điền0 hoặc suy diễn. Mean/sampleSD theo folds ở `paired_comparison.csv`; với một fold, SD chưa xác định. Threshold cố định probability>0.5.

OOF AUC tính trên pooled probabilities; có thể khác mean fold AUC khi thứ hạng/calibration thay đổi giữa các mô hình fold.

## Error overlap và probability average

{markdown(overlap)}

`raw_only` là trường hợp Raw đúng/Kinematic sai; `kin_only` là Raw sai/Kinematic đúng. Không coi corrected cases riêng lẻ là bằng chứng complementary value. Fusion duy nhất là xác suất **0.5 Kinematic +0.5 Raw**, đã định trước; không fit weights, gate hoặc stacking trên outer predictions. Tất cả probability mức participant ở `phase4_lite/oof_predictions.csv`.

{markdown(delta[delta.metric.isin(['f1_macro','auc'])])}

Paired bootstrap500 lần dùng chung participant resamples; trained models cố định, không tính uncertainty do refit/adaptive development. CI trên partial coverage chỉ áp dụng nhóm held-out đã hoàn tất. Không dùng kết quả này để chọn lại training recipe.

## Thời gian, bộ nhớ và resume

{markdown(forecast_table)}

{markdown(costs)}

Chi phí các fit chưa có OOF được ghi riêng, không bỏ khỏi accounting và không điền metrics giả:

{markdown(pending_table)}

Ước lượng cập nhật từ các fold thực sự đã hoàn tất (không dùng outer scores để đổi recipe):

{markdown(empirical_table)}

Benchmark dùng tám development-Train people, đầy đủ recordings; dự báo theo window count, scaling tới215 Train, safety factor1.5. Chi phí lịch sử Phase2 giữ riêng, không nhầm với thời gian đo mới. `training_seconds` gồm lưu resume checkpoint mỗi optimizer update; `evaluation_and_replay_seconds` gồm outer inference và replay checkpoint. Tensor allocated/reserved peak khác với tổng process memory trong nvidia-smi. Pilot benchmarks không phải outer fits.

Dự báo nhỏ có thể lệch vì warm-up, cache và kích thước các bag; thời gian actual của fold và session là số dùng cho accounting. Bảng forecast không phải thời gian đã tiêu tốn.

Mỗi phiên tối đa120 phút; mặc định110, có reserve cho ghi file/report. Checkpoint gồm weights, optimizer, CPU/CUDA/Python/NumPy RNG, permutation và cursor. Ngắt ở ranh giới optimizer update, tiếp tục đúng phần còn lại. Mọi epoch giữ toàn bộ cached samples, không silently truncate. Các phiên trong `phase4_lite/sessions/`; các fold dở trong `fits/*/status.json`.

{markdown(session_table)}

Log lỗi I/O của lần chạy đầu được giữ trong `session_current.stderr.log` và `io_failure_recovery.json`. Checkpoint trước lỗi được resume với cùng training signature; retry chỉ thay thao tác ghi file. Thời hạn kết thúc ban đầu giữ nguyên. Cumulative fit timers lấy từ checkpoint; tổng thời gian session cũng bao gồm công việc dở trước lỗi và startup. Hai loại thời gian không được cộng lặp.

Tiếp tục trong phiên khác: `.venv/Scripts/python.exe scripts/session_raw_temporal_v2_lite.py --minutes 110`. Wrapper bổ sung retry cho atomic checkpoint replace khi Windows tạm khóa file; code toán học và training signature được giữ nguyên. Không tự tạo phiên thứ hai sau khi chạm budget. Nếu có phục hồi do lỗi I/O, thời hạn kết thúc ban đầu vẫn được giữ bằng `--deadline-unix`.

## Khuyến nghị nghiên cứu

{recommendation}

Ưu tiên hoàn tất các folds của protocol lite đã khóa trước khi chi thêm cho nested search hoặc kiến trúc mới. Nếu Raw hoặc fixed-average không chứng minh lợi ích trên tập OOF đã hoàn tất, chưa có cơ sở chi ngân sách cho full nested fusion. Chỉ cân nhắc nested CV sau một tín hiệu ổn định cần kiểm chứng bằng Train-only selection và một ngân sách mới. SSL recovery mới là pilot; chưa có downstream comparison để khuyến nghị full SSL. Không mở thêm architecture grid từ kết quả outer folds hiện tại.

Bảo toàn artifacts cũ: đã kiểm tra {preservation['existing_files_checked']} paths và {preservation['protected_references_checked']} protected references bằng SHA. Baseline Kinematic OOF được dùng nguyên trạng. Báo cáo/code/results mới nằm riêng; không push hoặc merge tự động.
'''
    (OUT/'PHASE4_LITE_REPORT_VI.md').write_text(text,encoding='utf-8',newline='\n')
    write(LITE/'validation.json',dict(completed_fold_checks_passed=True,fold_checks=validation,
        all_requested_folds_complete=all_complete,phase3_complete=False,research_complete=False,
        protected_artifacts_unchanged=True,outer_test_fusion_weights_fitted=False))
    print('LITE REPORT',status,flush=True)


if __name__=='__main__':main()
