from . import physics, display
import numpy as np
import numpy.random as rand
import pandas as pd
import pickle
import warnings


class Simulation():
   '''
   A script for atomistic simulations of toy polymers. Each polymer is a linear string of beads
   (CH2 units) with no side chains. Bonds are springs and nonbonded atoms interact via a 12-6
   Lennard-Jones potential.
   
   Four methods of atomistic simulation are implemented: 

      (1) Steepest descent structural relaxation / energy minimization.

      (2) Constant temperature ('NVT') dynamics with a Langevin thermostat. 

      (3) Hamiltonian (aka `constant energy' or `NVE') molecular dynamics.

      (4) Metropolis Monte Carlo ('MMC') stochastic model.

   Paul Tangney, Ayham Al-Saffar 2024
   '''

   def __init__(self, n_atoms: int, starting_temperature: float, microstructure: pd.DataFrame = None,
                microstructure_logging_interval: int = 100) -> None:
      '''
      Initializes internal microstate, physics, and display variables.
      Units are in electron volts, femto-seconds, Angstroms, and Kelvin.

      Parameters
      ----------
      n_atoms : int
         Number of atoms in the simluated carbon chain.

      starting_temperature : int
         Temperature used when initiating atom velocities.

      microstructure : pd.DataFrame
         Dataframe specifying the intitial positions of each atom. Dataframe must have n_atoms rows
         and the columns 'x', 'y', 'z'. If not given, a straight linear chain is created.

      microstructure_logging_interval : int
         How many steps are run between each time the microstrucutre is saved. This can be reduced
         to as low as 10 without significantly impacting run speed, however this significantly
         increases the size of the saved simulation file. 
      '''
      self.n_atoms = n_atoms
      self.microstructure_logging_interval = microstructure_logging_interval
      self.is_being_displayed = False
      self.time_step = 0.8
      self.mass = 1451.0 # 14 AMU (CH2 Mr) in eV fs^2/Å^2

      #force parameters taken from https://doi.org/10.1063/1.476826 
      self.sigma = 4.5 #equilibrium Lennard Jones atomic seperation
      self.epsilon = 0.00485678 #Lennard Jones coefficient
      self.bond_length = 1.53 #equilibrium neighbour bond length
      self.spring_constant = 15.18

      #increases initial Boltzmann distribution velocities when generating a new microstate as some
      #of this energy is converted to potential energy. Its value should be 2.0 for a harmonic potential.
      T_factor = 1.5 
      self.kB = 8.617333262e-05
      self.target_kT = starting_temperature * self.kB
      self.velocity_sigma = np.sqrt(self.target_kT/self.mass) #variance of Boltzmann distrubution velocities
      self.velocities = rand.normal(loc=0.0, scale=np.sqrt(T_factor)*self.velocity_sigma, size=(self.n_atoms, 3))

      self.forces = np.zeros(shape=(self.n_atoms, 3))
      if microstructure is None:
         self.positions = np.zeros(shape=(self.n_atoms, 3))
         self.positions[:, 1] = np.linspace(start=0.0, stop=(self.n_atoms-1)*self.bond_length, num=self.n_atoms)
      else:
         assert microstructure.shape[0] == self.n_atoms
         assert all([col in microstructure.columns for col in ['x', 'y', 'z']])
         self.positions = microstructure[['x', 'y', 'z']].values
      
      self.positions -= np.mean(self.positions, axis=0, keepdims=True) #centre molecule
      self.velocities -= np.mean(self.velocities, axis=0, keepdims=True) #remove overall molecule movement

      #list with a dict {step: microstructure} for each run
      self.microstructures = [{0: pd.DataFrame(self.positions, columns=['x', 'y', 'z']).copy()}]
      self.run = 0
      data_structure = {'run': int, 'type': str, 'n_steps': int, 'T': float, 'KE': float, 'PE_bonding': float,
                        'PE_non_bonding': float, 'PE_total': float, 'F_rms': float, 'L_end_to_end': float}
      self.run_data = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in data_structure.items()})
      self._log_run_data(run_type='init', n_steps=0, temperature=starting_temperature)
      self.step_data = {} #{run: dataframe}


   def relax_run(self, n_steps: int, step_size: float = 0.01) -> None:
      '''
      Structural relaxation through steepest descent energy minimisation. This is a non-physical simulation
      where the molecule is modelled as not experiencing any thermal forces. Each atom moves in the direction
      that minimizes is potential energy the most, which is the net (bonding + non-bonding) force vector. 

      Parameters
      ----------
      n_steps : int
         Number of steps of steepest descent energy minimisation.

      step_size: float
         How far each atom should move with each step. The total movement vector is the relax_step_size * force
         vector. 0.01 is a good starting point, but could be ~3x larger for first few steps and ~10x smaller
         for the last few steps. Values larger than 0.03 risks creating force values too high to store as steps
         become unrealistically large.
      '''
      self.run += 1
      self.microstructures.append({0: pd.DataFrame(self.positions, columns=['x', 'y', 'z'])})
      step_data = []
      for step in range(n_steps):
         F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
         F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
         self.positions += (F_nb + F_b) * step_size
         step_data.append({
            'step': step,
            'PE_bonding': PE_b,
            'PE_non_bonding': PE_nb,
            'PE_total': PE_b + PE_nb,
            'F_rms': np.mean((F_b + F_nb)**2) ** 0.5,
            'L_end_to_end': np.sum((self.positions[0] - self.positions[-1]) ** 2) ** 0.5,
         })
         self._logging_step(step, step_data, run_type='relax', n_steps=n_steps, temperature=0.0)

      self.step_data[self.run] = pd.DataFrame(step_data)
      self._log_run_data(run_type='relax', n_steps=n_steps, temperature=0.0)
      if self.is_being_displayed: self.dashboard.reset(self)

   def NVT_run(self, n_steps: int, temperature: float, gamma: float = 0.005, integrator: str = 'OVRVO'):
      '''
      Constant number of atoms, volume, and temperature (NVT) simulation with a Langevin thermostat. Simple
      model that approximates the physical influence of a solute bath at a given temperature. It adds a drag
      and random force onto the energy minimization force.

      Parameters
      ----------
      n_steps : int
         Number of NVT steps to perform.

      temperature: float
         Temperature of simulation. A higher temperature leads to higher solute forces and so velocities. Note
         the actual temperature of the simulation at each step (dictated by atom velocities) may vary slightly.
         
      gamma: float
         Solvent interaction strength. Higher values lead to higher solute drag and random solute forces. 0 is
         no solvent, 0.001 is weak interaction, 0.005 is medium interaction, and 0.1 is strong interaction

      integrator: str
         How forces are intergrated over time. Either 'OVRVO' or 'Verlet'. Verlet is a well established 
         deterministic integrator that can be used to update positions and velocites after discrete timesteps.
         OVRO (See https://doi.org/10.1021/jp411770f) is a stochastic generalisation of Verlet that satisfies
         more of the criteria of an ideal integrator. 
      '''
      assert integrator in ('OVRVO', 'Verlet'), f'{integrator} integrator not supported'
      self.run += 1
      self.microstructures.append({0: pd.DataFrame(self.positions, columns=['x', 'y', 'z'])})
      step_data = []
   
      self.target_kT = temperature * self.kB
      self.velocity_sigma = np.sqrt(self.target_kT/self.mass) #variance of Boltzmann distrubution velocities

      F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
      F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
      a = np.exp(-gamma / (self.time_step*2)) #OVRVO constant
      b = np.sqrt(1 - a**2) #OVRVO constant
      for step in range(n_steps):
         if integrator == 'OVRVO':
            self.velocities = self.velocities*a + rand.standard_normal((self.n_atoms, 3))*self.velocity_sigma*b
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)
            self.positions += self.velocities*self.time_step
            F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
            F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)
            self.velocities = self.velocities*a + rand.standard_normal((self.n_atoms, 3))*self.velocity_sigma*b
         elif integrator == 'Verlet':
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)
            self.positions += self.velocities*self.time_step
            F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
            F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)

         KE_total = physics.get_kintetic_energy(self.velocities, self.mass)
         T_actual = physics.get_temperature(KE_total, self.n_atoms)
         step_data.append({
            'step': step,
            'T_actual': T_actual,
            'PE_total': PE_b + PE_nb,
            'KE_total': KE_total,
            'F_rms': np.mean((F_b + F_nb)**2) ** 0.5,
            'L_end_to_end': np.sum((self.positions[0] - self.positions[-1]) ** 2) ** 0.5,
         })
         self._logging_step(step, step_data, run_type='NVT', n_steps=n_steps, temperature=temperature)
      
      self.step_data[self.run] = pd.DataFrame(step_data)
      self._log_run_data(run_type='NVT', n_steps=n_steps, temperature=temperature)
      if self.is_being_displayed: self.dashboard.reset(self)

   def NVE_run(self, n_steps: int, temperature: float = None, gamma: float = 0.005, integrator: str = 'OVRVO'):
      '''
      Constant number of atoms, volume, and energy (NVE) simulation. Used for modelling molecules in the
      gas phase or finding an acceptable simulation timestep. Larger timesteps speed up the simulation
      but will eventually break conservation of energy if too large. https://doi.org/10.1021/jp411770f
      details different techniques for integrating the simulation forces over time.

      Parameters
      ----------
      n_steps : int
         Number of NVE steps to perform.

      temperature: float
         If given, resets the atom velocities to follow a boltzmann distribtuion at the
         new tempearture. Should be given if this run is not preceded by an NVT run at
         the desired temperature.

      gamma: float
         Solvent interaction strength. Higher values lead to higher solute drag and random solute forces. 0 is
         no solvent, 0.001 is weak interaction, 0.005 is medium interaction, and 0.1 is strong interaction

      integrator: str
         How forces are intergrated over time. Either 'OVRVO' or 'Verlet'. Verlet is a well established 
         deterministic integrator that can be used to update positions and velocites after discrete timesteps.
         OVRO (See https://doi.org/10.1021/jp411770f) is a stochastic generalisation of Verlet that satisfies
         more of the criteria of an ideal integrator. 
      '''
      assert integrator in ['OVRVO', 'Verlet'], f'integrator {integrator} not supported'
      self.run += 1
      self.microstructures.append({0: pd.DataFrame(self.positions, columns=['x', 'y', 'z'])})
      step_data = []

      if temperature is None:
         temperature = self.run_data.loc[self.run_data['run'] == (self.run-1), 'T'].values[0] #previous temperature
      else:
         self.target_kT = temperature * self.kB
         self.velocity_sigma = np.sqrt(self.target_kT/self.mass) #variance of Boltzmann distrubution velocities
         self.velocities = rand.normal(loc=0.0, scale=self.velocity_sigma, size=(self.n_atoms, 3))
      
      F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
      F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
      a = np.exp(-gamma / (self.time_step*2)) #OVRVO constant
      b = np.sqrt(1 - a**2) #OVRVO constant
      for step in range(n_steps):
         if integrator == 'OVRVO':
            self.velocities = self.velocities*a + rand.standard_normal((self.n_atoms, 3))*self.velocity_sigma*b
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)
            self.positions += self.velocities*self.time_step
            F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
            F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)
            self.velocities = self.velocities*a + rand.standard_normal((self.n_atoms, 3))*self.velocity_sigma*b
         elif integrator == 'Verlet':
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)
            self.positions += self.velocities*self.time_step
            F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
            F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
            self.velocities += (F_b + F_nb)*self.time_step / (2*self.mass)

         KE_total = physics.get_kintetic_energy(self.velocities, self.mass)
         step_data.append({
            'step': step,
            'PE_total': PE_b + PE_nb,
            'KE_total': KE_total,
            'F_rms': np.mean((F_b + F_nb)**2) ** 0.5,
            'L_end_to_end': np.sum((self.positions[0] - self.positions[-1]) ** 2) ** 0.5,
         })
         self._logging_step(step, step_data, run_type='NVE', n_steps=n_steps, temperature=temperature)
      
      self.step_data[self.run] = pd.DataFrame(step_data)
      self._log_run_data(run_type='NVE', n_steps=n_steps, temperature=temperature)
      if self.is_being_displayed: self.dashboard.reset(self)

   def MMC_run(self, n_steps: int, temperature: float, random_scale: float = 0.05):
      '''
      Metropolis Monte Carlo (MMC) Simulation. Randomly displaces atoms by a small amount and accepts the new
      structure if it reduces total potential energy or has a chance of accepting the new structure ∝ exp(-ΔPE)
      if it increases total potential energy. Un-physical but useful for sampling a variety of fairly low energy
      microstructures for thermodynamic property prediction. 

      Parameters
      ----------
      n_steps : int
         Number of MMC steps to perform.

      temperature: float
         Temperature of simulation. A higher temperature means higher thermal energy, increasing tha chance
         of an atom moving to a new random position.

      random_scale: float 
         Size limit of random displacement at each step. Will be +/- bond_length * random_scale / 2 in each direction.
      '''
      self.run += 1
      self.microstructures.append({0: pd.DataFrame(self.positions, columns=['x', 'y', 'z'])})
      step_data = []

      self.target_kT = temperature * self.kB
      self.velocity_sigma = np.sqrt(self.target_kT/self.mass) #variance of Boltzmann distrubution velocities

      energy_tracker = physics.PotentialEnergyTracker(self.positions, self.epsilon, self.sigma,
                                                    self.bond_length, self.spring_constant)
      PE = energy_tracker.get_total_potential_energy()
      max_displacement_size = self.bond_length * random_scale
      atom_indexes = rand.randint(low=0, high=self.n_atoms-1, size=n_steps)
      displacements = rand.uniform(low=-max_displacement_size/2, high=max_displacement_size/2, size=(n_steps, 3))
      for step in range(n_steps):
         is_displacement_accepted = False
         PE_change = energy_tracker.test_displacement(atom_indexes[step], displacements[step])
         if PE_change < 0 or np.exp(-PE_change / self.target_kT) > rand.uniform(low=0, high=1):
            is_displacement_accepted = True
            energy_tracker.accept_last_displacement()
            PE += PE_change
            self.positions[atom_indexes[step]] += displacements[step]
         
         step_data.append({
            'step': step,
            'displacement_accepted': is_displacement_accepted,
            'PE_total': PE,
            'L_end_to_end': np.sum((self.positions[0] - self.positions[-1]) ** 2) ** 0.5,
         })
         self._logging_step(step, step_data, run_type='MMC', n_steps=n_steps, temperature=temperature)
      
      self.step_data[self.run] = pd.DataFrame(step_data)
      self._log_run_data(run_type='MMC', n_steps=n_steps, temperature=temperature)
      if self.is_being_displayed: self.dashboard.reset(self)
   
   def display(self, live_display_interval: int = None):
      '''
      Create an interactive dashboard that displays how the molecule's conformation and physical properties
      change throughout the simulation. The dashboard can only be displayed in Jupyter Notebook. The dashboard
      will update live if this method is called before a simulation run is started.

      Parameters
      ----------
      live_display_interval : int
         How many steps are run between each live display update. Defaults to the microstructure logging
         interval but a larger interval will significantly speed up longer live-displayed runs. Note MMC
         runs have a 5x longer interval due to their higher run speed.
      '''
      self.live_display_interval = self.microstructure_logging_interval
      if live_display_interval is not None:
         assert live_display_interval % self.microstructure_logging_interval == 0,\
               f'please select a display interval that is a multiple of {self.microstructure_logging_interval}'
         self.live_display_interval = live_display_interval

      try:
         shell = get_ipython().__class__.__name__
         if shell == 'ZMQInteractiveShell': #when in Jupyter Notebook
            self.is_being_displayed = True
            self.dashboard = display.SimulationDashboard(self)
            self.dashboard.display(self)
         else:
            warnings.warn(f'This functionality is only available in a Jupyter Notebook, not a {shell} shell')
      except NameError:
         warnings.warn('This functionality is only available in a Jupyter Notebook')

   def save(self, path: str) -> None:
      '''
      Save simulation object as a btye file.
   
      Parameters
      ----------
      path : str
         File location where Simulation object will be stored.  
      '''
      if self.is_being_displayed:
         dashboard = self.dashboard
         self.dashboard = None #the dashboard instance cannot be saved but can be recreated at any time
      with open(path, mode='wb') as file:
         pickle.dump(self, file)
      if self.is_being_displayed:
         self.dashboard = dashboard

   def _log_run_data(self, run_type: str, n_steps: int, temperature: float) -> None:
      '''logs the final state of molecule after each run and stores it in the self.run_data dataframe'''
      F_b, PE_b = physics.get_bonding_interactions(self.positions, self.bond_length, self.spring_constant)
      F_nb, PE_nb = physics.get_non_bonding_interactions(self.positions, self.epsilon, self.sigma)
      current_state = {
         'run': self.run,
         'type': run_type,
         'n_steps': n_steps,
         'T': temperature,
         'KE': physics.get_kintetic_energy(self.velocities, self.mass),
         'PE_bonding': PE_b,
         'PE_non_bonding': PE_nb,
         'PE_total': PE_b + PE_nb, 
         'F_rms': np.mean((F_b + F_nb)**2) ** 0.5,
         'L_end_to_end': np.sum((self.positions[0] - self.positions[-1]) ** 2) ** 0.5, 
      }
      self.run_data = pd.concat([self.run_data, pd.DataFrame([current_state])])
   
   def _logging_step(self, step: int, step_data: list[dict], run_type: str, n_steps: int, temperature: float) -> None:
      '''Saves microstructure and updates display at appropriate steps'''
      if step % self.microstructure_logging_interval == 0: 
         self.positions -= np.mean(self.positions, axis=0, keepdims=True) #centre molecule
         self.microstructures[self.run][step] = pd.DataFrame(self.positions, columns=['x', 'y', 'z']).copy()
      
      if self.is_being_displayed == False:
         return
      live_display_interval = 5 * self.live_display_interval if run_type == 'MMC' else self.live_display_interval
      if step % live_display_interval == 0: 
         self.step_data[self.run] = pd.DataFrame(step_data)
         self.dashboard.live_update(self, step, run_type, n_steps, temperature)

def load_simulation(path: str) -> Simulation:
   '''
   Load in a simulation object from a file. For safety reasons, only load files you have created.
   
      Parameters
      ----------
      path : str
         Simulation file location to be read in. 
   '''
   with open(path, mode='rb') as file:
      return pickle.load(file)