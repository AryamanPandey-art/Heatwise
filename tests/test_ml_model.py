import unittest
from ml_model import get_trained_predictor, ThermalMLPredictor
import simulation
import pandas as pd
from heatwise import Workload

class TestMLModel(unittest.TestCase):
    def test_telemetry_generation(self):
        df = simulation.generate_synthetic_training_data(n_samples=50)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        self.assertIn("post_workload_temp", df.columns)

    def test_model_training_and_evaluation(self):
        predictor = ThermalMLPredictor(n_estimators=10, max_depth=5)
        metrics = predictor.train()
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        
    def test_prediction_shape(self):
        predictor = ThermalMLPredictor(n_estimators=10, max_depth=5)
        predictor.train()
        sim = simulation.DataCenterSimulation()
        rack = sim.racks[1]
        pred_temp, temp_delta = predictor.predict_rack_placement(rack, Workload(gpu_demand_pct=50.0, intensity="High"))
        self.assertIsInstance(pred_temp, float)
        self.assertIsInstance(temp_delta, float)
        
    def test_feature_generation(self):
        sim = simulation.DataCenterSimulation()
        rack = sim.racks[1]
        wl = Workload(workload_type="AI Training", intensity="High")
        from ml_model import build_prediction_features, FEATURE_COLUMNS
        df = build_prediction_features(rack, wl)
        self.assertEqual(list(df.columns), FEATURE_COLUMNS)
        self.assertNotIn("post_workload_temp", df.columns) # No leakage
        self.assertEqual(df["workload_AI_Training"].iloc[0], 1)

if __name__ == '__main__':
    unittest.main()
