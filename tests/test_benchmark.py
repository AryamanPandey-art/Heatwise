import unittest
from heatwise import run_batch_simulation_benchmark

class TestBenchmark(unittest.TestCase):
    def test_benchmark_execution(self):
        comp_df, metrics_trk, timeline_d = run_batch_simulation_benchmark("Normal Operation", 5, 42)
        
        self.assertIsNotNone(comp_df)
        self.assertIn("Round Robin", metrics_trk)
        self.assertIn("Least Loaded", metrics_trk)
        self.assertIn("HeatWise", metrics_trk)
        
        self.assertEqual(len(timeline_d), 5)

if __name__ == '__main__':
    unittest.main()
