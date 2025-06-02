import imperial_materials_simulation as ims
import time
import pathlib as pl
import numpy as np

sim = ims.Simulation(n_atoms=22, starting_temperature=400)
times = []

for i in range(5):
    start = time.perf_counter()
    sim.relax_run(n_steps=200_000)
    end = time.perf_counter()
times.append(end-start)

home_path = pl.Path(__file__).parent
with open(home_path/'relax_results.txt', mode='a') as file:
    file.write(f'global vars repeated {np.mean(times)} +/-{np.std(times):.3e}\n')