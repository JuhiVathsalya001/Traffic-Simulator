import time

class FixedTimeTrafficLight:
    def __init__(self, green=5, yellow=2, red=5):
        # Durations in seconds
        self.states = [
            ("GREEN", green),
            ("YELLOW", yellow),
            ("RED", red)
        ]

    def run(self, cycles=3):
        print(f"Starting simulation for {cycles} cycles...\n")
        for i in range(cycles):
            print(f"--- Cycle {i+1} ---")
            for state_name, duration in self.states:
                print(f"Status: {state_name} ({duration}s)")
                # Pause the program to simulate real-time passing
                time.sleep(duration) 
            print("-" * 15)

# Example: Green for 10s, Yellow for 3s, Red for 10s
simulator = FixedTimeTrafficLight(green=10, yellow=3, red=10)
simulator.run(cycles=2)
