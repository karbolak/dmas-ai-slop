"""Catch lost feed conditions, broken pairing, or an incorrect paired contrast."""
import tempfile
import unittest
from pathlib import Path
from experiments.feed_capacity_robustness import run_experiment, summarize
from experiments.alpha_volume_sweep import load_config
import pandas as pd


class FeedCapacityTest(unittest.TestCase):
    def test_small_real_sweep_is_paired_and_repeatable(self):
        cfg = load_config(Path('configs/feed_capacity_robustness.yaml'))
        cfg['seeds']['replicates'] = 2
        cfg['network'].update(n_agents=20, mean_in_degree=3)
        cfg['simulation']['iterations'] = 2000
        with tempfile.TemporaryDirectory() as directory:
            first = run_experiment(cfg, Path(directory) / 'a')
            second = run_experiment(cfg, Path(directory) / 'b')
        self.assertEqual(len(first), 36)
        self.assertEqual(first.groupby(['replicate', 'feed_size']).size().tolist(), [6]*6)
        self.assertTrue((first.groupby('replicate')['simulation_seed'].nunique() == 1).all())
        self.assertTrue((first['total_activations'] == 2001).all())
        pd.testing.assert_frame_equal(first.drop(columns='elapsed_seconds'), second.drop(columns='elapsed_seconds'))

    def test_contrast_pairs_worlds_before_averaging(self):
        frame = pd.DataFrame([
            dict(replicate=i, feed_size=1, synthetic_fraction=.05,
                 activity_multiplier=m, exposure_minus_production=d)
            for i, m, d in [(0,3,.1), (1,3,.2), (1,15,.1), (0,15,-.1)]
        ])
        _, paired = summarize(frame)
        self.assertAlmostEqual(paired.iloc[0]['mean'], -.15)
        self.assertEqual(paired.iloc[0]['n'], 2)


if __name__ == '__main__':
    unittest.main()
