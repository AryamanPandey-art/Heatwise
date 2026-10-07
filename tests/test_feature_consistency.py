import unittest
import pandas as pd
import numpy as np
import simulation
import ml_model
import heatwise

class TestFeatureConsistency(unittest.TestCase):
    def test_training_recirculation_consistency(self):
        df = simulation.generate_synthetic_training_data(n_samples=10)
        self.assertIn("recirculation_factor", df.columns)
        self.assertTrue(all(df["recirculation_factor"] >= 0.92))
        self.assertTrue(all(df["recirculation_factor"] <= 1.35))
        
    def test_inference_recirculation_consistency(self):
        sim = simulation.DataCenterSimulation(scenario="Normal Operation")
        wl = heatwise.Workload("AI Training", 3, 50, 50)
        
        rack_1 = sim.racks[1]
        rack_2 = sim.racks[2]
        
        rack_1["recirculation_factor"] = 1.00
        rack_2["recirculation_factor"] = 1.25
        
        feat_1 = ml_model.build_prediction_features(rack_1, wl)
        feat_2 = ml_model.build_prediction_features(rack_2, wl)
        
        self.assertEqual(feat_1.iloc[0]["recirculation_factor"], 1.00)
        self.assertEqual(feat_2.iloc[0]["recirculation_factor"], 1.25)
        
        # Ensure outputs differ on the new feature
        self.assertNotEqual(feat_1.iloc[0]["recirculation_factor"], feat_2.iloc[0]["recirculation_factor"])

if __name__ == '__main__':
    unittest.main()
