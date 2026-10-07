import unittest
from heatwise import Workload, HeatWiseDecisionEngine
from simulation import DataCenterSimulation

class TestHeatWise(unittest.TestCase):
    def setUp(self):
        self.sim = DataCenterSimulation("Thermal Hotspot")
        self.engine = HeatWiseDecisionEngine()

    def test_workload_has_memory_and_duration(self):
        wl = Workload("AI Training")
        self.assertTrue(hasattr(wl, "memory_demand_pct"))
        self.assertTrue(hasattr(wl, "duration_s"))

    def test_valid_rack_selection(self):
        wl = Workload("Light Inference", gpu_demand_pct=10.0, memory_demand_pct=10.0)
        result = self.engine.evaluate_placement(self.sim, wl)
        self.assertIsNotNone(result["recommended_rack"])

    def test_insufficient_memory_rejection(self):
        wl = Workload("Memory Hog", gpu_demand_pct=10.0, memory_demand_pct=400.0) # Impossible demand
        result = self.engine.evaluate_placement(self.sim, wl)
        self.assertTrue(result["recommended_rack"]["is_rejected"])
        self.assertIn("Memory capacity", result["recommended_rack"]["rejection_reason"])
        
    def test_insufficient_gpu_rejection(self):
        wl = Workload("GPU Hog", gpu_demand_pct=400.0, memory_demand_pct=10.0) # Impossible demand
        result = self.engine.evaluate_placement(self.sim, wl)
        self.assertTrue(result["recommended_rack"]["is_rejected"])
        self.assertIn("GPU capacity", result["recommended_rack"]["rejection_reason"])

if __name__ == '__main__':
    unittest.main()
