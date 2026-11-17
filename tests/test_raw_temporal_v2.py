"""Participant isolation and known-label fixture integrity for recovery work."""
import sys
import unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import numpy as np
import torch
from raw_temporal_v2_common import development,balanced_tiny,seed_all
from synthetic_raw_temporal_v2 import fixture,ComponentClassifier,tensors,TESTS


class RecoveryTests(unittest.TestCase):
    def test_development_split_excludes_designated_outer_test(self):
        ids,y,schema,fit,val,test=development()
        self.assertEqual((len(fit),len(val),len(test)),(143,72,43))
        self.assertFalse(set(fit)&set(val));self.assertFalse(set(fit)&set(test));self.assertFalse(set(val)&set(test))

    def test_tiny_sets_are_balanced_nested_train_subsets(self):
        ids,y,_,fit,_,test=development()
        for size in (8,16):
            chosen=balanced_tiny(size)
            self.assertEqual(len(chosen),size);self.assertEqual(int(y[chosen].sum()),size//2)
            self.assertTrue(set(chosen)<=set(fit));self.assertFalse(set(chosen)&set(test))
        self.assertTrue(set(balanced_tiny(8))<=set(balanced_tiny(16)))

    def test_order_labels_have_identical_mean_std_features(self):
        a=fixture('event_order',16,99)
        np.testing.assert_array_equal(a['features'][::2],a['features'][1::2])
        self.assertTrue(np.any(a['x'][::2]!=a['x'][1::2]))

    def test_timing_fixture_changes_real_dt_with_same_sample_patterns(self):
        a=fixture('physical_time',8,101)
        channels=[i for i in range(21) if i!=6]
        np.testing.assert_array_equal(a['x'][::2][:,:,channels],a['x'][1::2][:,:,channels])
        self.assertTrue(np.all(a['physical_dt'][::2]!=a['physical_dt'][1::2]))

    def test_synthetic_heldout_generators_are_independent(self):
        for kind in TESTS:
            a=fixture(kind,8,1001);b=fixture(kind,8,2001)
            self.assertFalse(np.array_equal(a['x'],b['x']))
            np.testing.assert_array_equal(a['y'],b['y'])

    def test_synthetic_components_ignore_absent_bag_tokens(self):
        seed_all(42);a=fixture('weak_bag',4,99);t=tensors(a)
        for family in ('cnn','tcn','hierarchical'):
            model=ComponentClassifier(family).to(t['x'].device).eval()
            with torch.no_grad():
                expected=model(*(t[k] for k in ('x','length','present','meta')))
                changed=t['x'].clone();changed[~t['present']]=12345
                actual=model(changed,t['length'],t['present'],t['meta'])
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)


if __name__=='__main__':unittest.main()
