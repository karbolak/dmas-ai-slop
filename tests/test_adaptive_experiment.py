import tempfile
import unittest
from pathlib import Path
from experiments.adaptive_producer_sweep import condition_grid, run_experiment
from experiments.alpha_volume_sweep import load_config


class AdaptiveExperimentTest(unittest.TestCase):
    def test_design_is_unique_and_covers_matched_controls(self):
        cfg=load_config(Path('configs/adaptive_producers.yaml'))
        grid=condition_grid(cfg)
        self.assertEqual(len(grid),36)
        self.assertEqual(len(set(grid)),36)
        self.assertEqual({p for a,f,m,p in grid},{'baseline','random','adaptive'})
        self.assertEqual(sum(a==1 for a,f,m,p in grid),24)

    def test_real_small_run_keeps_attempts_and_pairing(self):
        cfg=load_config(Path('configs/adaptive_producers.yaml'))
        cfg['seeds']['replicates']=2
        cfg['network'].update(n_agents=40,mean_in_degree=4)
        cfg['simulation']['iterations']=2000
        cfg['design']=[dict(feed_sizes=[1],fractions=[.1],activity_multipliers=[2.])]
        with tempfile.TemporaryDirectory() as directory:
            frame=run_experiment(cfg,Path(directory)/'run')
        self.assertEqual(len(frame),6)
        self.assertTrue((frame.total_activations==2001).all())
        self.assertTrue((frame.synthetic_creation_attempts>=frame.synthetic_posts).all())
        self.assertTrue((frame.groupby('replicate').simulation_seed.nunique()==1).all())


if __name__=='__main__': unittest.main()
