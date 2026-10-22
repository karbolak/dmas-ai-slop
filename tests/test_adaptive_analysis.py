import unittest
import pandas as pd
from analysis.analyze_adaptive_producers import paired_estimate


class PairedAnalysisTest(unittest.TestCase):
    def test_policy_pairs_match_world_id_instead_of_row_order(self):
        df=pd.DataFrame([
            {'replicate':0,'policy':'baseline','value':.1},
            {'replicate':1,'policy':'adaptive','value':.15},
            {'replicate':1,'policy':'baseline','value':.2},
            {'replicate':0,'policy':'adaptive','value':.3},
        ])
        result=paired_estimate(df,'value','adaptive','baseline')
        self.assertAlmostEqual(result['mean'],.075)
        self.assertEqual(result['n'],2)
        with self.assertRaises(ValueError):
            paired_estimate(df.iloc[:-1],'value','adaptive','baseline')

class RecordedHashTest(unittest.TestCase):
    def test_missing_run_machine_files_are_reported_but_changed_files_fail(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from analysis.analyze_adaptive_producers import verify_recorded_hashes
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'source.py';file.write_text('original')
            missing=Path(directory)/'other-machine.so'
            digest=hashlib.sha256(file.read_bytes()).hexdigest()
            checked,unavailable=verify_recorded_hashes({str(file):digest,str(missing):'recorded'})
            self.assertEqual(checked,1)
            self.assertEqual(unavailable,[str(missing)])
            file.write_text('changed')
            with self.assertRaises(ValueError):
                verify_recorded_hashes({str(file):digest})


if __name__=='__main__': unittest.main()
