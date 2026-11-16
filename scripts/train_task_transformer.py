"""Nested, participant-independent evaluation of small Kinematic token models."""
import argparse
import copy
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from task_transformer_common import ROOT, OUT, CONFIG, FAMILIES, METRICS, data, cache, device, score, initialize, training_signature, write, sha, factory
from diagramo.task_transformer import TaskPreprocessor, KinematicTaskNet, deterministic_seed, architecture
from diagramo.kinematic_xai import feature_mapping, train_only_task_importance


def sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


def predict(model, preprocessor, X, present, dev, return_attention=False):
    model.eval()
    Z = torch.tensor(preprocessor.transform(X), device=dev)
    present = np.asarray(present, dtype=bool)
    if present.ndim == 1:
        present = np.broadcast_to(present, (len(X), 14)).copy()
    mask = torch.tensor(present, device=dev)
    sync(dev)
    began = time.perf_counter()
    with torch.no_grad():
        if return_attention:
            logits, attention, pool = model(Z, mask, return_attention=True)
            attention = np.stack([v.cpu().numpy() for v in attention]) if attention else np.empty((0,))
            pool = pool.cpu().numpy()
        else:
            logits = model(Z, mask)
        prob = torch.sigmoid(logits).cpu().numpy().astype(float)
    sync(dev)
    return prob, dict(seconds=time.perf_counter()-began,
          **({'attention': attention, 'pool_weights': pool} if return_attention else {}))


def fit(X, y, candidate, family, present, seed, dev, X_val=None, y_val=None, epochs=None, augmentations=None):
    deterministic_seed(seed)
    pre = TaskPreprocessor().fit(X)
    Z = torch.tensor(pre.transform(X), device=dev)
    target = torch.tensor(y, dtype=torch.float32, device=dev)
    mask = torch.tensor(present, dtype=torch.bool, device=dev).expand(len(X), -1).clone()
    aug = torch.tensor(np.stack([pre.transform(a) for a in augmentations]), device=dev) if augmentations is not None else None
    val = torch.tensor(pre.transform(X_val), device=dev) if X_val is not None else None
    val_target = torch.tensor(y_val, dtype=torch.float32, device=dev) if y_val is not None else None
    val_mask = torch.tensor(present, dtype=torch.bool, device=dev).expand(len(X_val), -1) if X_val is not None else None
    model = KinematicTaskNet(**architecture(candidate, family)).to(dev)
    cfg = CONFIG['training']
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss, best_epoch, best_state, bad = float('inf'), 0, None, 0
    history = []
    sync(dev)
    began = time.perf_counter()
    for epoch in range(1, (epochs or cfg['max_epochs']) + 1):
        model.train()
        ordering = torch.randperm(len(X), device=dev)
        for start in range(0, len(X), cfg['batch_size']):
            indices = ordering[start:start+cfg['batch_size']]
            batch, m = Z[indices].clone(), mask[indices].clone()
            if aug is not None:
                use = torch.rand(len(indices), device=dev) < cfg['augmentation_probability']
                kind = torch.randint(len(aug), (len(indices),), device=dev)
                batch[use] = aug[kind[use], indices[use]]
            # Same training-only task dropout in every neural family.
            drop = (torch.rand(m.shape, device=dev) < cfg['task_dropout_probability']) & m
            proposed = m & ~drop
            m = torch.where(proposed.any(dim=1, keepdim=True), proposed, m)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(batch, m), target[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Z, mask)
            row = dict(epoch=epoch, train_loss=float(criterion(logits, target)),
                       train_accuracy=float(((logits > 0) == target.bool()).float().mean()))
            if val is not None:
                vl = model(val, val_mask)
                val_loss = float(criterion(vl, val_target))
                row.update(validation_loss=val_loss,
                           validation_accuracy=float(((vl > 0) == val_target.bool()).float().mean()))
                if val_loss < best_loss - 1e-5:
                    best_loss, best_epoch = val_loss, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
        history.append(row)
        if val is not None and epoch >= cfg['minimum_epochs'] and bad >= cfg['patience']:
            break
    if val is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    sync(dev)
    return model, pre, history, dict(best_epoch=int(best_epoch), epochs_ran=len(history),
           train_seconds=time.perf_counter()-began, parameters=sum(p.numel() for p in model.parameters()),
           peak_cuda_bytes=int(torch.cuda.max_memory_allocated()) if dev == 'cuda' else 0)


def checkpoint(path, model, pre, mask, train_ids, validation_ids, candidate, family, seed, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state={k: v.detach().cpu() for k,v in model.state_dict().items()},
                   architecture=model.config, preprocessor=pre, present=np.asarray(mask),
                   train_ids=np.asarray(train_ids), validation_ids=np.asarray(validation_ids),
                   candidate=candidate, family=family, seed=seed, metadata=metadata), path)


def load_model(path, dev):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    model = KinematicTaskNet(**bundle['architecture']).to(dev)
    model.load_state_dict(bundle['state'])
    model.eval()
    return model, bundle


def safe_ranking(X, y, ids, schema, fit_indices, forbidden_indices, folder, outer=None):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'task_importance.csv'
    if target.exists():
        info = json.loads((folder / 'provenance.json').read_text())
        assert info['fit_ids'] == ids[fit_indices].tolist()
        assert info['forbidden_ids'] == ids[forbidden_indices].tolist()
        return pd.read_csv(target)
    if outer is not None:
        existing = ROOT / 'results/kinematic_xai/train_only' / f'outer_fold_{outer}'
        provenance = json.loads((existing / 'provenance.json').read_text())
        assert provenance['outer_train_ids'] == ids[fit_indices].tolist()
        assert provenance['input_X_sha256'] == hashlib.sha256(X[fit_indices].tobytes()).hexdigest()
        assert provenance['input_y_sha256'] == hashlib.sha256(y[fit_indices].tobytes()).hexdigest()
        ranks = pd.read_csv(existing / 'task_importance.csv')
        origin = str(existing.relative_to(ROOT))
        inner_splits = provenance['inner_splits']
    else:
        mapping = feature_mapping(schema['tasks'], schema['features_per_task'])
        inner_splits = []
        def record(f, tr, va, pipe, exp):
            inner_splits.append(dict(inner=f, train_ids=ids[fit_indices][tr].tolist(),
                                     validation_ids=ids[fit_indices][va].tolist()))
        ranks, phi, folds = train_only_task_importance(X[fit_indices], y[fit_indices], ids[fit_indices], mapping,
                          factory, seed=42, forbidden_ids=ids[forbidden_indices], save_split=record)
        np.savez_compressed(folder / 'cross_fitted.npz', phi=phi, ids=ids[fit_indices], inner_fold=folds)
        origin = 'recomputed exclusively on current inner Train'
    ranks.to_csv(target, index=False)
    write(folder / 'provenance.json', dict(fit_ids=ids[fit_indices].tolist(), forbidden_ids=ids[forbidden_indices].tolist(),
          source=origin, inner_splits=inner_splits, no_outer_test_attributions_used=True))
    return ranks


def task_mask(family, candidate, schema, ranking=None):
    present = np.ones(14, dtype=bool)
    if family == 'without_text_tasks':
        present[[schema['tasks'].index(t) for t in ('copy', 'dictation')]] = False
    elif family == 'xai_selected':
        selected = ranking.sort_values(['rank','task']).head(candidate['top_k']).task
        present = np.array([t in set(selected) for t in schema['tasks']])
    return present


def run_outer(f, dev, signature):
    from task_transformer_common import features_for
    X, y, ids, schema, assignments, original, splits = data()
    bank = cache() if (OUT / 'cache/features.npz').exists() else {'raw_clean': X}
    tr, te = splits[f]
    destination = OUT / 'folds' / f'fold_{f}'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'complete.json').exists():
        assert json.loads((destination / 'complete.json').read_text())['signature'] == signature
        print('REUSE OUTER', f, flush=True)
        return
    began = time.perf_counter()
    inner_splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X[tr], y[tr]))
    rankings = []
    for j, (itr, iva) in enumerate(inner_splits):
        rankings.append(safe_ranking(X, y, ids, schema, tr[itr], np.r_[tr[iva], te],
                         OUT / 'train_only' / f'outer_{f}' / f'inner_{j}'))
    outer_ranking = safe_ranking(X, y, ids, schema, tr, te, OUT / 'train_only' / f'outer_{f}' / 'final_train', outer=f)
    search_rows, inner_predictions = [], []
    execution_order = [name for name in FAMILIES if name not in ('motor_normalized','train_augmented')] + ['motor_normalized','train_augmented']
    for family in execution_order:
        if family in ('motor_normalized','train_augmented') and isinstance(bank,dict):
            while not (OUT / 'cache/features.npz').exists():
                print('WAITING raw feature bank; completed raw-family inner searches are saved',flush=True)
                time.sleep(15)
            bank=cache()
        feature = features_for(family, bank)
        for ci, candidate in enumerate(CONFIG['candidates']):
            for j, (itr, iva) in enumerate(inner_splits):
                target = destination / 'inner' / f'{family}_{ci}_{j}.json'
                if target.exists():
                    row = json.loads(target.read_text())
                    assert row['signature'] == signature
                    search_rows.append(row)
                    continue
                present = task_mask(family, candidate, schema, rankings[j])
                augment = [bank['train_' + n][tr[itr]] for n in CONFIG['nuisances']] if family == 'train_augmented' else None
                model, pre, history, cost = fit(feature[tr[itr]], y[tr[itr]], candidate, family, present,
                          42, dev, X_val=feature[tr[iva]], y_val=y[tr[iva]], augmentations=augment)
                prob, timing = predict(model, pre, feature[tr[iva]], present, dev)
                p = OUT / 'checkpoints' / f'outer_{f}/inner/{family}_{ci}_{j}.pt'
                row = dict(fold=f, family=family, candidate=ci, inner=j, seed=42,
                     signature=signature, train_ids=ids[tr[itr]].tolist(), validation_ids=ids[tr[iva]].tolist(),
                     present=present.tolist(), **score(y[tr[iva]], prob), **cost,
                     checkpoint=p.relative_to(ROOT).as_posix())
                checkpoint(p, model, pre, present, ids[tr[itr]], ids[tr[iva]], candidate, family, 42, row)
                row['checkpoint_sha256'] = sha(p)
                write(target, row)
                write(target.with_name(target.stem + '_history.json'), history)
                pd.DataFrame(dict(participant_id=ids[tr[iva]], label=y[tr[iva]], probability=prob,
                             prediction=(prob > .5).astype(int))).to_csv(target.with_suffix('.oof.csv'), index=False)
                search_rows.append(row)
        print('INNER DONE', f, family, flush=True)
    search = pd.DataFrame(search_rows)
    search.to_csv(destination / 'inner_search.csv', index=False)
    averaged = search.groupby(['family','candidate'], sort=False).agg(
            f1_macro=('f1_macro','mean'), auc=('auc','mean'), parameters=('parameters','first'),
            epochs=('best_epoch','median')).reset_index()
    chosen = {}
    for family in FAMILIES:
        row = averaged[averaged.family == family].sort_values(['f1_macro','auc','parameters','candidate'], ascending=[False,False,True,True]).iloc[0]
        chosen[family] = dict(candidate=int(row.candidate), epochs=max(1,int(row.epochs)), inner_f1=float(row.f1_macro), inner_auc=float(row.auc))
    overall = averaged.sort_values(['f1_macro','auc','parameters','candidate','family'], ascending=[False,False,True,True,True]).iloc[0]
    write(destination / 'selection.json', dict(families=chosen, nested_selected_family=str(overall.family),
                 nested_selected_candidate=int(overall.candidate), no_outer_test_labels_used=True))
    outer_rows = []
    # Main per-family results, repeated seeds and predeclared matched controls.
    jobs = [(family, int(chosen[family]['candidate']), seed, family, chosen[family]['epochs'])
             for family in FAMILIES for seed in CONFIG['repeat_seeds']]
    for family, ci, name in [('transformer',0,'matched_transformer_no_type'),('transformer',1,'matched_transformer_type'),('task_mlp',0,'matched_task_mlp')]:
        epoch = int(averaged[(averaged.family == family) & (averaged.candidate == ci)].iloc[0].epochs)
        jobs.append((family, ci, 42, name, max(1,epoch)))
    for family, ci, seed, label, epochs in jobs:
        target = destination / 'outer' / f'{label}_seed_{seed}.json'
        if target.exists():
            assert json.loads(target.read_text())['signature'] == signature
            outer_rows.append(json.loads(target.read_text()))
            continue
        candidate = CONFIG['candidates'][ci]
        present = task_mask(family, candidate, schema, outer_ranking)
        feature = features_for(family, bank)
        augment = [bank['train_' + n][tr] for n in CONFIG['nuisances']] if family == 'train_augmented' else None
        model, pre, history, cost = fit(feature[tr], y[tr], candidate, family, present, seed, dev, epochs=epochs, augmentations=augment)
        prob, timing = predict(model, pre, feature[te], present, dev)
        path = OUT / 'checkpoints' / f'outer_{f}/final/{label}_seed_{seed}.pt'
        row = dict(fold=f, model=label, family=family, seed=seed, candidate=ci, signature=signature,
             epochs=epochs, train_ids=ids[tr].tolist(), test_ids=ids[te].tolist(), present=present.tolist(),
             **cost, inference_seconds=timing['seconds'], **score(y[te], prob), checkpoint=path.relative_to(ROOT).as_posix())
        checkpoint(path, model, pre, present, ids[tr], ids[te], candidate, family, seed, row)
        row['checkpoint_sha256'] = sha(path)
        write(target, row)
        write(target.with_name(target.stem + '_history.json'), history)
        pd.DataFrame(dict(participant_id=ids[te], label=y[te], fold=f, model=label, seed=seed,
                         probability=prob, prediction=(prob > .5).astype(int))).to_csv(target.with_suffix('.oof.csv'), index=False)
        outer_rows.append(row)
    # Baseline seed 42 is loaded, never retrained; other seeds are sensitivity-only.
    for seed in CONFIG['repeat_seeds']:
        for exclude_text in (False, True):
            label = 'lightgbm_without_text' if exclude_text else 'lightgbm'
            columns = np.arange(1036)
            if exclude_text:
                columns = np.flatnonzero(np.repeat(task_mask('without_text_tasks', CONFIG['candidates'][0], schema),74))
            from evaluate import classifiers, build_pipeline
            if seed == 42 and not exclude_text:
                path = ROOT / 'results/models/kinematic/LightGBM' / f'fold_{f}.joblib'
                model = joblib.load(path)
            else:
                model = build_pipeline('LightGBM', classifiers(seed)['LightGBM']).fit(X[tr][:,columns],y[tr])
                path = OUT / 'checkpoints' / f'outer_{f}/{label}_seed_{seed}.joblib'
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path, compress=3)
            pred = model.predict(X[te][:,columns]); prob = model.predict_proba(X[te][:,columns])[:,1]
            if seed == 42 and not exclude_text:
                np.testing.assert_array_equal(pred, original.prediction.to_numpy()[te])
                np.testing.assert_allclose(prob, original.probability.to_numpy()[te], atol=1e-12, rtol=0)
            target = destination / 'outer' / f'{label}_seed_{seed}.json'
            write(target, dict(fold=f, model=label, seed=seed, **score(y[te],prob,pred),
                 columns=columns.tolist(), checkpoint=path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(path),
                 train_ids=ids[tr].tolist(), test_ids=ids[te].tolist(), original_baseline_reused=seed == 42 and not exclude_text))
            pd.DataFrame(dict(participant_id=ids[te],label=y[te],fold=f,model=label,seed=seed,
                             probability=prob,prediction=pred)).to_csv(target.with_suffix('.oof.csv'),index=False)
    write(destination / 'complete.json', dict(complete=True, signature=signature, outer_fold=f,
          inner_fits=len(search), neural_final_fits=len(jobs), seconds=time.perf_counter()-began))
    print(f'OUTER {f} COMPLETE: {len(search)} inner + {len(jobs)} neural final fits; {time.perf_counter()-began:.1f}s',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(6)))
    ap.add_argument('--pilot', action='store_true')
    args=ap.parse_args()
    initialize()
    dev=device()
    if args.pilot:
        X,y,ids,schema,assignment,oof,splits=data()
        tr,te=splits[0]
        model,pre,history,cost=fit(X[tr],y[tr],CONFIG['candidates'][0],'transformer',np.ones(14,bool),42,dev,epochs=10)
        write(OUT/'pilot.json', dict(device=dev,train_ids=ids[tr].tolist(),outer_test_not_evaluated=True,**cost))
        print('PILOT',cost,flush=True)
        return
    signature=training_signature()
    with threadpool_limits(limits=4):
        for f in args.folds:
            run_outer(f,dev,signature)


if __name__=='__main__': main()
