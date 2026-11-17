"""Truthful held-out waveforms, technical boundaries and model explanations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_temporal_v2_common import *
from diagramo.temporal.preprocessing import preprocess


def main():
    active=json.loads((OUT/'phase4_complementarity/xai/active_run.json').read_text());run=ROOT/active['run']
    assert json.loads((run/'validation.json').read_text())['complete']
    figdir=run/'figures';figdir.mkdir(exist_ok=True);manifest=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig,sources,description,extra=None):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(figdir/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        manifest.append(dict(name=name,description=description,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            extra=extra,outputs={e:dict(path=(figdir/(name+'.'+e)).relative_to(ROOT).as_posix(),sha256=sha(figdir/(name+'.'+e))) for e in ('png','svg')}));plt.close(fig)
    tasks=pd.read_csv(run/'task_occlusion.csv');fold=pd.read_csv(run/'task_importance_by_fold.csv')
    methods=sorted(tasks.model.unique());fig,axes=plt.subplots(1,len(methods),figsize=(6*len(methods),6))
    for ax,model in zip(axes,methods):
        a=fold[(fold.model==model)&(fold.seed==42)];mean=a.groupby('task').absolute_logit_delta.mean().sort_values();sd=a.groupby('task').absolute_logit_delta.std()
        ax.errorbar(mean.values,np.arange(len(mean)),xerr=sd.loc[mean.index],fmt='o',capsize=3)
        ax.set_yticks(range(len(mean)),mean.index);ax.set_xlabel('Mean absolute removal Δlogit ±SD across6 folds');ax.set_title(model+', primary seed42')
    save('task_importance',fig,[run/'task_importance_by_fold.csv'],'Task removal sensitivity. Fold SD is descriptive, not a CI from independent cohorts; attention weights are not treated as explanations.')
    stability=pd.read_csv(run/'reference_seed_randomization.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,column in zip(axes,['ig_reference_spearman','task_head_randomization_spearman']):
        values=stability[stability.seed==42][column].dropna();ax.hist(values,bins=np.linspace(-1,1,11),edgecolor='black')
        ax.set(xlim=(-1,1),xlabel=column,ylabel='Selected held-out participants',title=f'Valid comparisons n={len(values)}')
    save('reference_randomization',fig,[run/'reference_seed_randomization.csv'],'Selected-case diagnostic; high random-head similarity weakens a substantive interpretation. Missing/failed IG is excluded, never replaced by zero.')
    deletion=pd.read_csv(run/'deletion_faithfulness.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ranking in zip(axes,['signed_ig_support','occlusion_tested_subset_support']):
        a=deletion[(deletion.seed==42)&(deletion.attribution_rank_type==ranking)]
        if not len(a):ax.text(.5,.5,'No eligible cases',transform=ax.transAxes,ha='center');continue
        # Average random repeats within participant before averaging people.
        person=a.groupby(['participant_id','method','fraction'],as_index=False).original_prediction_confidence_logit_drop.mean()
        for method,group in person.groupby('method'):
            average=group.groupby('fraction').original_prediction_confidence_logit_drop.mean();ax.plot(average.index,average.values,marker='o',label=method)
        ax.set(xlabel='Fraction removed within eligible technical segments',ylabel='Original predicted-group confidence logit drop',title=ranking);ax.legend(fontsize=7)
    save('deletion_faithfulness',fig,[run/'deletion_faithfulness.csv'],'Primary-seed selected-case mean, random repeats averaged within participant. Correct and wrong predictions are both included; confidence faithfulness is not diagnostic validity.')
    cases=pd.read_csv(run/'selected_cases.csv');recordings=pd.read_csv(OLD/'audit/recordings.csv').set_index(['participant_id','task'])
    for case in cases.to_dict('records'):
        pid=case['participant_id'];folder=run/'cases'/f'{pid}_seed42';validation=json.loads((folder/'validation.json').read_text())
        task=validation['chosen_task_name'];source_key=recordings.loc[(pid,task),'source_key'];source=raw_svc_path(pid,source_key)
        values=np.loadtxt(source,skiprows=1);order=('x','y','time','pen_status','azimuth','tilt','pressure')
        record={name:values[:,j] for j,name in enumerate(order)};pre=preprocess(record);t=pre['time']-pre['time'][0]
        bad_clock=~pre['physical'][:,15].astype(bool);bad_clock[0]=False
        mapping=pd.read_csv(folder/'source_mapping.csv');chosen=mapping[mapping.task_index==validation['chosen_task']]
        coverage=np.zeros(len(t),dtype='int32')
        for row in chosen.itertuples():
            coverage[row.raw_start_index:row.raw_end_index]+=1
            assert abs(pre['time'][row.raw_start_index]-row.source_start_seconds)<1e-6
            assert abs(pre['time'][row.raw_end_index-1]-row.source_end_seconds)<1e-6
        assert (coverage==1).all(),'Attribution source-index mapping omits or duplicates original samples'
        boundaries=np.flatnonzero(pre['boundary']);extrema=[int(np.argmax(record['pressure']))]
        if pre['velocity_valid'].any():extrema.append(int(np.argmax(np.where(pre['velocity_valid'],pre['physical'][:,10],-np.inf))))
        shown=np.unique(np.r_[np.linspace(0,len(t)-1,min(6000,len(t))).astype(int),boundaries,np.maximum(boundaries-1,0),extrema])
        segments=pd.read_csv(folder/'segment_occlusion.csv');highlight=segments.sort_values(['prediction_support','segment_id'],ascending=[False,True]).head(3)
        fig=plt.figure(figsize=(14,14));grid=fig.add_gridspec(4,2);xy=fig.add_subplot(grid[0,0]);taskax=fig.add_subplot(grid[0,1])
        pressure=fig.add_subplot(grid[1,0]);speed=fig.add_subplot(grid[1,1]);pen=fig.add_subplot(grid[2,0]);effects=fig.add_subplot(grid[2,1]);igax=fig.add_subplot(grid[3,:])
        physical=pre['physical'];cursor=physical[shown,:2].copy();cursor[bad_clock[shown]]=np.nan
        xy.plot(cursor[:,0],cursor[:,1],color='#BBBBBB',linewidth=.4,label='Cursor path')
        ink=cursor.copy();ink[pre['pen'][shown]==0]=np.nan;xy.plot(ink[:,0],ink[:,1],color='#24506B',linewidth=.7,label='Pen-down')
        xy.set(xlabel='Relative x (source device units)',ylabel='Relative y (source device units)',title='(a) Original acquisition trajectory');xy.set_aspect('equal',adjustable='datalim');xy.legend(fontsize=7)
        taskrows=tasks[(tasks.participant_id==pid)&(tasks.model==active['target'])&(tasks.seed==42)].sort_values('absolute_logit_delta')
        taskax.barh(taskrows.task,taskrows.logit_delta,color=['#24506B' if v>=0 else '#BD6C24' for v in taskrows.logit_delta]);taskax.axvline(0,color='black',linewidth=.5)
        taskax.set(xlabel='Removal Δlogit (+ supports group1)',title='(b) Whole-task occlusion')
        pressure_line=record['pressure'].copy();pressure_line[bad_clock]=np.nan
        pressure.plot(t[shown],pressure_line[shown],color='#24506B',linewidth=.6);pressure.set(xlabel='Time since task start (s)',ylabel='Pressure (source units)',title='(c) Pressure; invalid-clock gaps retained')
        actual_speed=physical[:,10].copy();actual_speed[~pre['velocity_valid']]=np.nan
        speed.plot(t[shown],actual_speed[shown],color='#24506B',linewidth=.6);speed.set(xlabel='Time since task start (s)',ylabel='Speed (source units/s)',title='(d) Valid pen-down speed; gaps retained')
        pen_line=pre['pen'].astype('float64');pen_line[bad_clock]=np.nan
        pen.step(t[shown],pen_line[shown],where='post',color='#24506B',linewidth=.7);pen.plot(t[boundaries],np.full(len(boundaries),-.08),'|',color='#777777',markersize=4,label='Technical segment boundaries')
        pen.set(xlabel='Time since task start (s)',ylabel='Pen state',ylim=(-.15,1.15),title='(e) Pen status and boundaries');pen.legend(fontsize=7)
        effects.scatter(segments.source_start_seconds-pre['time'][0],segments.logit_delta,s=15,marker='o');effects.axhline(0,color='black',linewidth=.5)
        effects.set(xlabel='Segment start since task start (s)',ylabel='Removal Δlogit',title=f'(f) Tested technical segments {len(segments)}/{validation["total_chosen_task_segments"]}')
        colors=['#B3446C','#47956E','#9B7D24']
        for (_,segment),color in zip(highlight.iterrows(),colors):
            a,b=int(segment.raw_start_index),int(segment.raw_end_index)
            selected=shown[(shown>=a)&(shown<b)]
            if len(selected):xy.plot(physical[selected,0],physical[selected,1],color=color,linewidth=1.5,label=f'Segment {int(segment.segment_id)}')
            for ax in (pressure,speed,pen):ax.axvspan(t[a],t[b-1],color=color,alpha=.2)
        check=next(v for v in validation['ig_validations'] if v['reference']=='train_mean')
        sources=[source,folder/'source_mapping.csv',folder/'segment_occlusion.csv',run/'task_occlusion.csv',folder/'validation.json']
        if check['complete']:
            path=ROOT/check['attribution_path'];assert sha(path)==check['attribution_sha256']
            with np.load(path) as data:attribution=data['attribution']
            aligned=np.zeros(len(t),dtype='float64')
            for row in chosen.itertuples():aligned[row.raw_start_index:row.raw_end_index]=np.abs(attribution[row.patch_id,:,:row.valid_samples]).astype('float64').sum(0)
            aligned[bad_clock]=np.nan
            igax.plot(t[shown],aligned[shown],color='#24506B',linewidth=.6);sources.append(path)
            igax.set(ylabel='Absolute motor-contrast logit attribution / sample',xlabel='Actual source time since task start (s)',title='(g) Conditional motor IG; time, flags and hierarchy metadata held fixed')
        else:igax.text(.5,.5,'IG numerical validation failed; no substantive interpretation',transform=igax.transAxes,ha='center');igax.set_axis_off()
        xy.legend(fontsize=6)
        fig.suptitle(f'{pid} — {task} — held-out group1 p={validation["original_probability"]:.3f}, true group={validation["label"]}; {case["category"]}\nModel explanation, not a causal or clinical conclusion',fontsize=11)
        save('wave_'+pid,fig,sources,'Real held-out case. Colors mark largest original-prediction support among tested segments; not clinical motor events. Invalid derivatives shown as gaps.',
            extra=dict(raw_samples=len(t),display_samples=len(shown),display_policy='Uniform up to6000 plus all boundaries, pre-boundary samples and pressure/speed extrema; model keeps all samples',
                mapping_all_source_samples_covered_exactly_once=True,coordinate_units='Uncalibrated source device units',physical_time_source='Original SVC time, task-start offset only'))
    write(run/'figure_manifest.json',manifest)
    validation=json.loads((OUT/'phase4_complementarity/xai/validation.json').read_text());validation.update(figures_pending=False,figure_count=len(manifest),figure_manifest_sha256=sha(run/'figure_manifest.json'))
    write(OUT/'phase4_complementarity/xai/validation.json',validation)
    print('XAI FIGURES EXPORTED',len(manifest),'PNG/SVG pairs',flush=True)


if __name__=='__main__':main()
