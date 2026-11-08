"""Mechanism tests: baseline invariance, channel order, frozen adapter and controls."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagramo.channel_embedding import FixedChannelAdapter,channel_rgb,aggregate_channel_embeddings
from evaluate_embedding_ablation import select_from_inner,evaluate_split
from evaluate import classifiers
from threadpoolctl import threadpool_limits


class ChannelEmbeddingTests(unittest.TestCase):
    def test_original_stem_equals_channel_mean_and_loses_channel_order(self):
        torch.manual_seed(42)
        rgb=nn.Conv2d(3,5,3,bias=True).double()
        seven=nn.Conv2d(7,5,3,bias=True).double()
        with torch.no_grad():
            seven.weight.copy_(rgb.weight.sum(1,keepdim=True).repeat(1,7,1,1)/7)
            seven.bias.copy_(rgb.bias)
        x=torch.rand(2,7,9,9,dtype=torch.float64)
        torch.testing.assert_close(seven(x),rgb(x.mean(1,keepdim=True).repeat(1,3,1,1)))
        torch.testing.assert_close(seven(x),seven(x[:,[6,5,4,3,2,1,0]]))

    def test_group_adapter_is_frozen_and_uses_all_channels(self):
        adapter=FixedChannelAdapter()
        self.assertEqual(list(adapter.parameters()),[])
        weights=adapter.weight.flatten(1)
        torch.testing.assert_close(weights.sum(1),torch.ones(3))
        self.assertTrue(torch.all(weights.sum(0)>0))
        x=torch.arange(7,dtype=torch.float32).reshape(1,7,1,1)
        torch.testing.assert_close(adapter(x).flatten(),torch.tensor([.5,4.,4.]))

    def test_separate_channel_order_and_mean_control(self):
        x=torch.arange(14,dtype=torch.float32).reshape(2,7,1,1)
        rgb=channel_rgb(x)
        torch.testing.assert_close(rgb[:,0,0,0],torch.arange(14,dtype=torch.float32))
        features=torch.arange(42,dtype=torch.float32).reshape(14,3)
        concat,mean=aggregate_channel_embeddings(features,2)
        torch.testing.assert_close(concat,features.reshape(2,21))
        torch.testing.assert_close(mean,features.reshape(2,7,3).mean(1))
        swapped=features.reshape(2,7,3).flip(1).reshape(14,3)
        swapped_concat,swapped_mean=aggregate_channel_embeddings(swapped,2)
        self.assertFalse(torch.equal(concat,swapped_concat))
        torch.testing.assert_close(mean,swapped_mean)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError): FixedChannelAdapter()(torch.zeros(2,3,8,8))
        with self.assertRaises(ValueError): aggregate_channel_embeddings(torch.zeros(13,4),2)

    def test_selection_is_from_complete_inner_scores(self):
        rows=[]
        for mode in ['original','group_adapter','channel_mean','channel_concat']:
            for k in range(3):
                for reducer,value in [('none',.6),('pca_16',.7)]:
                    rows.append(dict(mode=mode,inner_fold=k,reducer=reducer,classifier='SVM',
                                     n_features=16,accuracy=value,f1_macro=value,auc=value))
        selections,_=select_from_inner(rows)
        self.assertEqual(selections['all_selected']['reducer'],'pca_16')
        self.assertEqual(selections['original_raw']['reducer'],'none')
        with self.assertRaises(ValueError): select_from_inner(rows[:-1])

    def test_actual_split_grid_uses_training_ids_and_complete_predictions(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(96,80)).astype('float32')
        x[:,79]=5
        x[0,0]=np.nan
        y=np.tile([0,1],48)
        ids=np.array([f'p{i}' for i in range(96)])
        tr=np.arange(64); te=np.arange(64,96)
        with threadpool_limits(limits=2):
            r=evaluate_split(x,y[tr],ids,tr,te,y[te],classifiers(42),True)
        self.assertEqual(r['bank']['fit_participants'],ids[tr].tolist())
        self.assertFalse(set(r['bank']['fit_participants'])&set(ids[te]))
        self.assertEqual(len(r['scores']),24)
        self.assertEqual(len(r['predictions']),24)
        for item in r['scores']:
            self.assertEqual(item['n_features'],r['reduction_metadata'][item['reducer']]['n_features'])
            p=r['predictions'][item['reducer']+'__'+item['classifier']]
            self.assertEqual(len(p['prediction']),32)
            self.assertTrue(np.isfinite(p['probability']).all())


if __name__=='__main__': unittest.main()
