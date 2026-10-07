import unittest
from simulation import DataCenterSimulation

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = DataCenterSimulation("Thermal Hotspot")
        self.rack = self.sim.racks[1]

    def test_workload_increases_power_and_temp(self):
        initial_power = self.rack["total_power_w"]
        initial_temp = self.rack["temperature"]
        
        # Simulate placement
        result = self.sim.simulate_placement(1, 50.0, 50.0, 1000.0)
        
        self.assertGreater(result["new_power_w"], initial_power)
        self.assertGreater(result["new_temp"], initial_temp)
        
    def test_memory_utilization_validity(self):
        self.assertIn("mem_capacity_gb", self.rack)
        self.assertIn("mem_available_pct", self.rack)
        self.assertGreaterEqual(self.rack["mem_available_pct"], 0.0)
        self.assertLessEqual(self.rack["mem_available_pct"], 100.0)

    def test_inlet_outlet_temperatures(self):
        self.assertIn("inlet_temp", self.rack)
        self.assertIn("outlet_temp", self.rack)
        self.assertGreater(self.rack["outlet_temp"], self.rack["inlet_temp"])

    def test_memory_usage_increases_on_placement(self):
        result = self.sim.simulate_placement(1, 50.0, 40.0, 1000.0)
        self.assertGreater(result["new_mem_util"], self.rack["avg_mem_util"])
        self.assertLessEqual(result["new_mem_util"], 100.0)

if __name__ == '__main__':
    unittest.main()
