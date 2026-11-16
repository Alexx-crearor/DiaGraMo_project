"""General paper-draft PNG/SVG figures and actual raw-signal explanations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch,Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve
from raw_temporal_common import *
from diagramo.temporal.preprocessing import preprocess

T=OUT/'tables';D=OUT/'figures';D.mkdir(exist_ok=True);manifest=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'rawtemporal42','savefig.facecolor':'white'})
def read(name):return pd.read_csv(T/(name+'.csv'))
def save(name,fig,sources,caption):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(D/(name+'.'+ext),dpi=300,bbox_inches='tight',metadata={'Date':None}if ext=='svg'else None)
    manifest.append(dict(name=name,caption=caption,publication_status='general paper draft; journal requirements not specified',dpi=300,
      independent_unit='participant, n=258; 6 folds; repeated seeds use the same cohort',uncertainty='Fold SD unless explicitly labelled; not confidence intervals',
      sources={p.relative_to(ROOT).as_posix():sha(p)for p in sources},outputs={ext:sha(D/(name+'.'+ext))for ext in ('png','svg')}));plt.close(fig)
def heat(ax,df,title,cmap='viridis',signed=False):
    A=df.to_numpy(float);limit=np.nanmax(abs(A))if np.isfinite(A).any()else 1.
    if limit==0:limit=1.
    cm=plt.get_cmap(cmap).copy();cm.set_bad('#b9b9b9');im=ax.imshow(np.ma.masked_invalid(A),aspect='auto',cmap=cm,vmin=-limit if signed else None,vmax=limit if signed else None)
    ax.set_xticks(range(len(df.columns)),df.columns,rotation=35,ha='right');ax.set_yticks(range(len(df)),df.index);ax.set_title(title);plt.colorbar(im,ax=ax,shrink=.8)

def example_figures():
    ids,y,schema,_,_=cohort();cases=read('selected_cases')
    for row in cases.itertuples():
        occlusion=read('task_occlusion');q=occlusion[(occlusion.participant_id==row.participant_id)&(occlusion.model=='hierarchical')&(occlusion.seed==42)]
        task=q.iloc[q.logit_delta.abs().to_numpy().argmax()].task;ti=schema['tasks'].index(task)
        records,meta,path=recording_map(row.participant_id,schema['tasks']);raw=records[task][1];pre=preprocess(raw);time=pre['time']-pre['time'][0];pressure=np.asarray(raw['pressure']);x=np.asarray(raw['x']);yy=np.asarray(raw['y'])
        source=raw_svc_path(row.participant_id,records[task][0])
        attribution=OUT/'xai/examples'/(row.participant_id+'_seed_42_zero_joint.npz')
        with np.load(attribution)as archive:ig_samples=archive['x']
        cache=np.load(OUT/'cache'/(row.participant_id+'.npz'))
        offset=sum(len(cache[f'n{j}'])for j in range(ti));inv=cache[f'm{ti}'];signal=np.zeros(len(time))
        for j,(start,end,*_)in enumerate(inv):
            start,end=int(start),int(end);signal[start:end]=ig_samples[offset+j,:,:end-start].sum(0)
        # Actual pen-down edges only; invalid time/pen transitions are not joined.
        valid=pre['velocity_valid'][1:];points=np.column_stack([x,yy]);edges=np.stack([points[:-1],points[1:]],1)[valid]
        colors=time[1:][valid];fig,axes=plt.subplots(4,1,figsize=(11,11));ax=axes[0]
        lc=LineCollection(edges,array=colors,cmap='viridis',linewidths=.6,rasterized=True);ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal',adjustable='datalim');plt.colorbar(lc,ax=ax,label='Recorded elapsed time (source seconds)');ax.set_xlabel('x (source coordinate units)');ax.set_ylabel('y (source coordinate units)')
        ax.set_title(f'{row.category}: {row.participant_id} / {task} — highest |task occlusion logit effect|; p={row.probability:.3f}')
        speed=np.where(pre['velocity_valid'],pre['physical'][:,10],np.nan);acc=np.where(pre['acceleration_valid'],np.hypot(pre['physical'][:,11],pre['physical'][:,12]),np.nan)
        bad=np.flatnonzero((pre['dt']<=0)|(pre['dt']>1.));bad=bad[bad>0]
        broken_time=np.insert(time,bad,np.nan)
        axes[1].plot(broken_time,np.insert(speed,bad,np.nan),color='#167c9b',lw=.6,label='Speed, valid pen-down intervals',rasterized=True);axes[1].set_ylabel('Coordinate units / source second');axes[1].legend(fontsize=8)
        twin=axes[2].twinx();axes[2].plot(broken_time,np.insert(pressure,bad,np.nan),color='#9e5f2a',lw=.6,label='Pressure',rasterized=True);twin.plot(broken_time,np.insert(pre['pen'].astype(float),bad,np.nan),color='#5f7060',alpha=.45,lw=.5,label='Pen state');axes[2].set_ylabel('Pressure (source units)');twin.set_ylabel('Pen-down=1 / pen-up=0');twin.set_yticks([0,1]);axes[2].legend(loc='upper left')
        axes[2].scatter(time[bad],pressure[bad],s=3,c='#ac433d',label='Clock/gap boundary',rasterized=True)
        axes[3].plot(broken_time,np.insert(signal,bad,np.nan),lw=.6,color='#954c91',rasterized=True);axes[3].scatter(time[bad],signal[bad],s=3,c='#ac433d',rasterized=True);axes[3].axhline(0,color='black',lw=.5);axes[3].set_ylabel('Signed sample-channel IG sum\n(positive-class logit)')
        for ax in axes[1:]:ax.set_xlabel('Recorded elapsed time (source seconds; not normalized)')
        for at in np.flatnonzero(pre['boundary']):
            for ax in axes[1:]:ax.axvline(time[at],color='grey',alpha=.08,lw=.4)
        save('case_'+row.category+'_raw_signals',fig,[T/'selected_cases.csv',T/'task_occlusion.csv',path,source,attribution],
          'Actual full source signals, no resampling; rasterized dense traces inside vector SVG. Invalid derivatives are gaps; pen state/boundaries retained. IG overlay includes sample-channel contributions only, time-metadata attribution remains separately stored. No clinical diagnosis inferred.')
        strokes=read('stroke_occlusion');s=strokes[(strokes.participant_id==row.participant_id)&(strokes.task==task)].sort_values('time_start')
        fig,axes=plt.subplots(2,1,figsize=(11,5.8));axes[0].bar(s.original_run_index,s.logit_delta,color=np.where(s.pen_state==1,'#25758b','#aaa18c'));axes[0].axhline(0,color='black',lw=.5);axes[0].set_ylabel('Full logit − run-occluded logit');axes[0].set_xlabel('Original pen-state run index (pen-up runs also retained)');axes[0].set_title(row.participant_id+' / '+task)
        axes[0].legend(handles=[Patch(color='#25758b',label='Pen-down run'),Patch(color='#aaa18c',label='Pen-up run')],fontsize=8)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].plot(time,acc,lw=.6,color='#427f71',rasterized=True);axes[1].set_xlabel('Recorded elapsed time (source seconds)');axes[1].set_ylabel('Acceleration magnitude\n(source coordinate units / source second squared)')
        save('case_'+row.category+'_stroke_effects',fig,[T/'stroke_occlusion.csv',path,source], 'Joint whole-run occlusion mapped to original run indices; pen-down/pen-up distinguished; acceleration is a recorded-signal calculation, not a diagnosis.')
        window_mass=np.abs(ig_samples[offset:offset+len(inv)]).sum((1,2));chosen=int(window_mass.argmax())
        description=read('behavioral_examples').set_index('participant_id').loc[row.participant_id]
        assert description.task==task and int(description.window_index)==chosen
        assert int(description.raw_start)==int(inv[chosen,0]) and int(description.raw_end_exclusive)==int(inv[chosen,1])
        left=int(inv[max(0,chosen-1),0]);right=int(inv[min(len(inv)-1,chosen+1),1]);selected=(time>=time[left])&(time<=time[right-1])
        fig,axes=plt.subplots(3,1,figsize=(11,6.5))
        axes[0].plot(broken_time,np.insert(speed,bad,np.nan),lw=.8,color='#167c9b');axes[0].set_ylabel('Valid speed')
        axes[1].plot(broken_time,np.insert(pressure,bad,np.nan),lw=.8,color='#9e5f2a');axes[1].set_ylabel('Source pressure')
        axes[2].plot(broken_time,np.insert(signal,bad,np.nan),lw=.8,color='#954c91');axes[2].scatter(time[bad],signal[bad],s=8,c='#ac433d');axes[2].axhline(0,color='black',lw=.5);axes[2].set_ylabel('Signed sample IG')
        start,end=inv[chosen,5]-pre['time'][0],inv[chosen,6]-pre['time'][0]
        for ax in axes:
            ax.set_xlim(time[left]-.001,time[right-1]+.001);ax.axvspan(start,end,color='#b9cdd8',alpha=.25);ax.set_xlabel('Recorded elapsed time (source seconds)')
        axes[0].set_title(row.participant_id+' / '+task+' — largest absolute sample-channel IG window; adjacent-window context')
        save('case_'+row.category+'_temporal_zoom',fig,[path,source,attribution,T/'behavioral_examples.csv'], 'Deterministic highest-absolute-IG window within the displayed task, independently matched to the descriptive table; not selected for visual appearance. No interpolation across invalid/gap intervals; red markers indicate clock boundaries. Narrow events are not diagnosed as tremor or disease.')
        components=read('ig_components');c=components[(components.participant_id==row.participant_id)&(components.seed==42)&(components.baseline=='zero_joint')&(components.kind=='trained')].iloc[0]
        increments=np.array([c.reference,c.sample_channel_sum,c.patch_time_sum,c.segment_time_sum,c.logit]);starts=np.r_[0,np.cumsum(increments[:3]),0]
        fig,ax=plt.subplots(figsize=(9,3.8));ax.bar(range(5),increments,bottom=starts,color=['#9c9c9c','#347f91','#b18140','#835f99','#426c52'])
        ends=starts+increments;low=min(0.,float(starts.min()),float(ends.min()));high=max(0.,float(starts.max()),float(ends.max()));span=max(high-low,.1)
        ax.set_ylim(low-.18*span,high+.22*span)
        for j,(v,b)in enumerate(zip(increments,starts)):ax.annotate(f'{v:+.4f}',(j,b+v),xytext=(0,3 if v>=0 else -3),textcoords='offset points',ha='center',va='bottom'if v>=0 else'top',fontsize=8)
        ax.set_xticks(range(5),['Reference logit','Sample-channel IG','Patch-time IG','Segment-time IG','Actual output logit']);ax.axhline(0,color='black',lw=.5);ax.set_ylabel('Positive-class logit');ax.set_title(f'{row.participant_id}: conditional IG decomposition; closure residual={c.residual:+.2e}')
        save('case_'+row.category+'_ig_components',fig,[T/'ig_components.csv',attribution], 'Reference plus signed sample/time-metadata IG approximates actual positive-class logit within validated tolerance. Structural task IDs, masks and segmentation held fixed; reference is mathematical, not a healthy recording.')

def main():
    audit=pd.read_csv(OUT/'audit/recordings.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    tasks=audit.task.drop_duplicates().tolist()
    for ax,column in zip(axes,['n_samples','duration_endpoint']):
        ax.boxplot([audit[audit.task==task][column]for task in tasks],tick_labels=tasks,showfliers=False);ax.set_yscale('log');ax.tick_params(axis='x',labelrotation=65);ax.set_ylabel(column+' (log10 axis)');ax.set_title('258 recordings per task; outliers omitted from box display only')
    save('01_raw_data_lengths',fig,[OUT/'audit/recordings.csv'],'All raw values remain in CSV; boxplots suppress outlier markers only. Time uses source seconds convention.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6));names=['duplicate Δt','negative Δt','tiny positive Δt','gaps >1s'];values=[audit.duplicate_dt.sum(),audit.negative_dt.sum(),audit.tiny_positive_dt.sum(),audit.large_dt_over_1s.sum()]
    axes[0].bar(names,values,color='#467e9b');axes[0].tick_params(axis='x',labelrotation=20);axes[0].set_ylabel('Intervals, full cohort')
    trade=pd.read_csv(OUT/'audit/window_tradeoff.csv').groupby('patch_size')[['patches','padded_cells']].sum();axes[1].bar(trade.index.astype(str),trade.patches);axes[1].set_xlabel('Maximum raw samples per window');axes[1].set_ylabel('Windows (all samples retained)')
    save('02_timestamp_and_window_audit',fig,[OUT/'audit/recordings.csv',OUT/'audit/window_tradeoff.csv'],'Fixed safety thresholds and computational window trade-off; no label-based window selection.')
    fig,ax=plt.subplots(figsize=(12,3.3));ax.axis('off');ax.set_xlim(-.015,1.02)
    labels=['Raw chronological samples\n21 timestamp-safe channels','Local CNN movement encoder\n128-sample masked windows','Stroke transformer\n≤16 windows per chunk','Task temporal transformer\n≤32 segments per chunk','14 task identities\n→ participant logit']
    for i,label in enumerate(labels):
        x=i*.2;ax.add_patch(FancyBboxPatch((x,.5),.18,.37,boxstyle='round,pad=.006',fc='#e8f0f7',ec='#406680'));ax.text(x+.09,.685,label,ha='center',va='center',fontsize=8)
        if i<4:ax.annotate('',(x+.2,.685),(x+.18,.685),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.25,'Physical time within recordings; task IDs across assessments, no chronological task positions.\nEvery source sample is covered; chunk summaries bound memory and limit direct long-range attention.',ha='center')
    ax.text(.5,.06,'Fixed outer6 folds; inner3 folds select configuration and early stopping; no outer-Test monitoring.',ha='center')
    save('03_raw_temporal_architecture',fig,[ROOT/'src/diagramo/temporal/hierarchical_model.py',ROOT/'configs/raw_temporal.json'],'Actual bounded hierarchy; CNN/TCN/flat Transformer are separate raw baselines, no handcrafted74-dimensional input.')
    primary=read('primary_ablation').set_index('model');names=['kinematic_lightgbm_reference','cnn','tcn','transformer','hierarchical','nested_selected'];fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,col in zip(axes.flat,['f1_macro','accuracy','auc','balanced_accuracy','sensitivity','specificity']):
        d=primary.loc[names];ax.barh(names,d['mean_'+col],xerr=d['std_'+col],color='#327c92',capsize=2,label='Fold mean ± SD');ax.scatter(d[col],names,c='#c76c24',s=18,zorder=4,label='Pooled OOF');ax.set_xlim(0,1);ax.invert_yaxis();ax.set_xlabel(col)
    axes.flat[0].legend(fontsize=7,loc='lower left');save('04_predictive_comparison',fig,[T/'primary_ablation.csv'],'Same258 participants/folds; reference LightGBM unchanged; nested_selected uses Train-only architecture choice.')
    ab=read('temporal_multiscale_ablation');fig,ax=plt.subplots(figsize=(10,4));ax.barh(ab.model,ab.mean_delta_f1_macro,xerr=ab.std_delta_f1_macro,capsize=3);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Paired fold Δ macro-F1 relative to full hierarchy (mean ± SD)');ax.invert_yaxis()
    save('05_time_and_multiscale_ablation',fig,[T/'temporal_multiscale_ablation.csv'],'Matched hierarchical dropout, Train-only epochs; no_position retains CNN order and clock-valued sample inputs; index_time retains integrity boundaries.')
    oof=read('oof_predictions');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in names[:-1]:
        d=oof[(oof.model==name)&(oof.seed==42)];fpr,tpr,_=roc_curve(d.label,d.probability);axes[0].plot(fpr,tpr,label=name)
    axes[0].plot([0,1],[0,1],'k--',lw=.6);axes[0].set_xlabel('False-positive rate');axes[0].set_ylabel('True-positive rate');axes[0].legend(fontsize=7)
    for i,name in enumerate(names[1:]):
        d=read('performance_summary');d=d[d.model==name].sort_values('seed');axes[1].scatter(i+np.linspace(-.13,.13,len(d)),d.f1_macro)
    axes[1].set_xticks(range(len(names)-1),names[1:],rotation=30,ha='right');axes[1].set_ylim(0,1);axes[1].set_ylabel('OOF macro-F1 (three seeds; same cohort)')
    save('06_roc_and_seed_sensitivity',fig,[T/'oof_predictions.csv',T/'performance_summary.csv'],'Pooled OOF ROC and seed sensitivity, not external validation or independent cohorts.')
    curves=read('learning_curves');fig,axes=plt.subplots(2,2,figsize=(11,6))
    for ax,name in zip(axes.flat,['cnn','tcn','transformer','hierarchical']):
        d=curves[(curves.model==name)&(curves.scope=='inner')].groupby('epoch')[['train_loss','validation_loss']].mean();ax.plot(d.index,d.train_loss,label='Online Train');ax.plot(d.index,d.validation_loss,label='Inner validation');ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('BCE')
    axes.flat[0].legend();save('07_learning_curves',fig,[T/'learning_curves.csv'],'Online Train uses dropout/changing weights; validation uses eval. Late epochs contain fewer surviving runs due to early stopping.')
    d=primary.loc[['cnn','tcn','transformer','hierarchical']];fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,col,label in zip(axes,['parameters','train_seconds_mean','peak_gpu_bytes'],['Trainable parameters','Final-fit training time (s)','Peak tensor memory per process (GiB)']):
        value=d[col]/1024**3 if col=='peak_gpu_bytes'else d[col];ax.bar(d.index,value);ax.set_ylabel(label);ax.tick_params(axis='x',labelrotation=25)
    save('08_resource_cost',fig,[T/'primary_ablation.csv'],'Final benchmark uses one sequential GPU worker; CPU XAI can share host resources. Descriptive workflow timings, not dedicated hardware benchmarking. Tensor peak excludes CUDA driver/context/reserved memory. All inputs covered, no crops.')
    importance=read('task_importance');a=importance[importance.seed==42].pivot(index='task',columns='model',values='mean_absolute_logit_delta');fig,ax=plt.subplots(figsize=(10,6));heat(ax,a,'Task occlusion: mean absolute positive-class logit effect')
    save('09_task_importance',fig,[T/'task_importance.csv'],'Model reliance measured by missing task tokens; no task selection or neural training weights from Kinematic SHAP.')
    st=read('fold_xai_stability');geo=read('representation_geometry_stability');fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,df,col,title in [(axes[0],st,'spearman','Task-rank fold stability (15 dependent fold pairs)'),(axes[1],geo,'spearman','Seed stability of held-out representation geometry')]:
        d=df.groupby('model')[col].agg(['mean','std']);ax.barh(d.index,d['mean'],xerr=d['std'],capsize=2);ax.set_xlim(-1,1);ax.set_xlabel(title+'; mean ± descriptive SD')
    save('10_explanation_and_geometry_stability',fig,[T/'fold_xai_stability.csv',T/'representation_geometry_stability.csv'],'Distance-rank geometry is rotation/translation invariant; pairwise distances and fold pairs are dependent.')
    pert=read('temporal_perturbation_summary');fig,ax=plt.subplots(figsize=(11,4));ax.barh(pert.kind,pert.delta_f1_macro_mean,xerr=pert.delta_f1_macro_std,capsize=3);ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Paired fold Δ macro-F1 (mean ± SD)')
    save('11_order_and_missing_segments',fig,[T/'temporal_perturbation_summary.csv'],'Complete pen-state runs reordered; internal movement retained but global timing can be OOD. Missing segments use masks, not artificial zero movement.')
    sanity=read('ig_baseline_seed_randomization');faith=read('faithfulness');fig,axes=plt.subplots(1,2,figsize=(12,4))
    d=sanity.groupby('comparison').patch_spearman.agg(['mean','std']);axes[0].barh(d.index,d['mean'],xerr=d['std'],capsize=2);axes[0].set_xlim(-1,1);axes[0].set_xlabel('Example patch IG rank correlation; mean ± SD')
    for name,d in faith.groupby('baseline'):axes[1].scatter(d.task_spearman,d.temporal_spearman,label=name)
    axes[1].set_xlim(-1,1);axes[1].set_ylim(-1,1);axes[1].set_xlabel('Task IG vs task occlusion Spearman');axes[1].set_ylabel('Temporal IG vs block occlusion Spearman');axes[1].legend(fontsize=7)
    save('12_ig_faithfulness_and_sanity',fig,[T/'ig_baseline_seed_randomization.csv',T/'faithfulness.csv'],'Documented TP/TN/FP/FN example set only; randomization/baseline sensitivity are diagnostics, not clinical validation.')
    att=read('attention_supplementary').groupby(['query_task','key_task']).attention.mean().unstack('key_task');fig,ax=plt.subplots(figsize=(10,8));heat(ax,att,'Supplementary task attention — not attribution')
    save('13_attention_supplementary',fig,[T/'attention_supplementary.csv'],'Mean task attention, not automatically a faithful explanation.')
    pairs=read('paired_comparisons').set_index('model').loc[names[1:]];fig,ax=plt.subplots(figsize=(10,4));ax.barh(pairs.index,pairs.corrected,color='#348a78',label='Corrected');ax.barh(pairs.index,-pairs.harmed,color='#bd5b4e',label='Harmed');ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Participants relative to protected LightGBM');ax.legend()
    save('14_paired_corrected_harmed',fig,[T/'paired_comparisons.csv'],'Different error cases are descriptive complementarity hypotheses, not evidence from a trained fusion.')
    cm=read('confusion_matrices');fig,axes=plt.subplots(1,3,figsize=(9,3))
    for ax,name in zip(axes,['kinematic_lightgbm_reference','hierarchical','nested_selected']):
        d=cm[cm.model==name].pivot(index='label',columns='prediction',values='n');ax.imshow(d,cmap='Blues',vmin=0,vmax=143)
        for i in (0,1):
            for j in (0,1):ax.text(j,i,str(d.loc[i,j]),ha='center',va='center')
        ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('Prediction');ax.set_ylabel('True group');ax.set_title(name,fontsize=8)
    save('15_confusion_matrices',fig,[T/'confusion_matrices.csv'],'Seed42 participant-level OOF counts; original threshold0.5.')
    example_figures();write(D/'manifest.json',manifest);print('RAW TEMPORAL FIGURES',len(manifest),flush=True)

if __name__=='__main__':main()
