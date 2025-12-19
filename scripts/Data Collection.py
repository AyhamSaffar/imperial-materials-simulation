import imperial_materials_simulation as ims
import concurrent.futures as cf #Python standard multiprocessing library 

def sample_workflow(temperature: int) -> None:
    #reduce size of simulation files by increasing the microstructure logging interval
    simulation = ims.Simulation(n_atoms=22, starting_temperature=temperature, microstructure_logging_interval=1_000)
    simulation.NVT_run(n_steps=100_000, temperature=temperature)
    simulation.MMC_run(n_steps=500_000, temperature=temperature)
    simulation.to_file(f'simulation {temperature}k.ims') 

if __name__ == '__main__':
    with cf.ProcessPoolExecutor() as executor: #only works in a .py file
        for temperature in range(100, 901, 100):
            executor.submit(sample_workflow, temperature)
