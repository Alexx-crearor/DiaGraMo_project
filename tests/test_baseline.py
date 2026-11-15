"""Read-only regression checks for the restored original reproduction."""
import itertools
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from diagramo import features
from diagramo.branches import kinematic, moment, gaf, cwt
from evaluate import classifiers, build_pipeline
from prepare import TASKS, canonical


class BaselineTests(unittest.TestCase):
    def test_original_formula_entrypoints(self):
        self.assertIs(kinematic.extract, features.kinematic)
        self.assertIs(gaf.transform, features.gaf)
        self.assertIs(cwt.transform, features.cwt)
        signals = np.load(ROOT / 'data/derived/signals512.npy', mmap_mode='r')[0, 0]
        g = gaf.transform(np.stack([features.resample(a, 224) for a in signals]))
        w = cwt.transform(signals)
        self.assertEqual(g.shape, (7, 224, 224))
        self.assertEqual(w.shape, (7, 224, 224))
        self.assertTrue(np.isfinite(g).all() and np.isfinite(w).all())

    def test_kinematic_against_saved_real_recordings(self):
        saved = np.load(ROOT / 'data/derived/kinematic.npz')
        schema = json.loads((ROOT / 'data/derived/schema.json').read_text())
        self.assertEqual(TASKS, schema['tasks'])
        self.assertEqual(list(features.CHANNELS), schema['channels'])
        for i in (0, len(saved['ids']) // 2, len(saved['ids']) - 1):
            raw = json.loads((ROOT / 'data/clean_data' / (saved['ids'][i] + '_merged.json')).read_text(encoding='utf-8'))
            records = {canonical(key): value['data'] for key, value in raw.items() if canonical(key) in TASKS}
            extracted = []
            for task in TASKS:
                values, names = kinematic.extract(records[task])
                self.assertEqual(names, schema['features_per_task'])
                extracted.extend(values)
            np.testing.assert_array_equal(extracted, saved['X'][i])

    def test_moment_preserves_api_and_tensor(self):
        x = object()
        embeddings = object()
        def frozen_model(*, x_enc):
            self.assertIs(x_enc, x)
            return SimpleNamespace(embeddings=embeddings)
        self.assertIs(moment.extract(frozen_model, x), embeddings)
        self.assertEqual(moment.DIMENSION_PER_PARTICIPANT, 14336)

    def test_ten_cached_representations(self):
        base = np.load(ROOT / 'data/derived/kinematic.npz')
        dimensions = {'kinematic': 1036, 'moment': 14336}
        for branch in ('gaf', 'cwt'):
            for encoder, dim in [('resnet50', 2048), ('efficientnet_b4', 1792), ('vit_b_16', 768), ('vgg16', 4096)]:
                dimensions[branch + '_' + encoder] = 14 * dim
        for name, dim in dimensions.items():
            with np.load(ROOT / 'data/derived' / (name + '.npz')) as d:
                np.testing.assert_array_equal(d['ids'], base['ids'])
                np.testing.assert_array_equal(d['y'], base['y'])
                self.assertEqual(d['X'].shape, (258, dim))
                self.assertTrue(np.isfinite(d['X']).all())

    def test_fixed_classifier_parameters_and_pipeline(self):
        settings = json.loads((ROOT / 'results/cv/settings.json').read_text())
        self.assertEqual(settings['seed'], 42)
        for name, model in classifiers(42).items():
            self.assertEqual(json.loads(json.dumps(model.get_params(), default=str)), settings['classifiers'][name])
            pipe = build_pipeline(name, model)
            self.assertEqual([type(s).__name__ for _, s in pipe.steps[:3]],
                             ['SimpleImputer', 'VarianceThreshold', 'StandardScaler'])
            self.assertEqual(model.get_params(), classifiers(42)[name].get_params())
            if name == 'SVM':
                self.assertEqual(type(pipe.steps[3][1]).__name__, 'RBFKernelTransformer')
                self.assertEqual(pipe.steps[-1][1].kernel, 'precomputed')
                self.assertEqual(model.kernel, 'rbf')

    def test_all_eleven_original_early_fusions(self):
        selected = {'Kinematic': 'kinematic', 'MOMENT': 'moment',
                    'GAF': 'gaf_vit_b_16', 'CWT': 'cwt_efficientnet_b4'}
        count = 0
        for n in (2, 3, 4):
            for combo in itertools.combinations(selected, n):
                count += 1
                name = '+'.join(combo)
                expected = [selected[k] for k in combo]
                sample = np.concatenate([np.load(ROOT / 'data/derived' / (k + '.npz'))['X'][:1] for k in expected], axis=1)
                for clf in classifiers(42):
                    r = json.loads((ROOT / 'results/cv' / (name + '__' + clf + '.json')).read_text())
                    self.assertEqual(r['feature_keys'], expected)
                    self.assertEqual(r['n_features'], sample.shape[1])
        self.assertEqual(count, 11)


if __name__ == '__main__':
    unittest.main()
