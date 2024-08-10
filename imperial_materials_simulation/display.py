'''
Methods for creating interactive visualisations of the simulated molecule and its measurements
within Jupyter Notebook using ipywidgets. 
'''
import ipywidgets as ipy
import py3Dmol
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from IPython.display import display, clear_output

class SimulationDashboard():
    '''
    Class for creating inline Jupter Notebook visualisations of how the measurements and microstructure of simulated
    molecules vary live with time. This is built on top of the library's Simulation object.
    '''

    def __init__(self, sim) -> None:
        '''Initiates internal methods, attributes, and dashboard widgets'''
        #TODO replace matplotlib with plotly for speed
        matplotlib.use('module://ipympl.backend_nbagg')
        self.sim = sim

        self.mol_viewer = py3Dmol.view(width=300, height=300)
        with plt.ioff():
            self.fig, self.left_ax = plt.subplots(figsize=(6, 3))
            self.fig.tight_layout(pad=2)
            self.right_ax = self.left_ax.twinx()
            self.line = self.right_ax.axvline(x=0.5, color='black', linestyle='--')
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.footer_visible = False
        #TODO remove resize corner

        self.observers_enabled = False
        self.run_slider = ipy.IntSlider(value=0, min=0, max=0, description='Run', orientation='vertical')
        self.table_widget = ipy.Output(layout=ipy.Layout(height='200px', overflow='auto'))
        with self.table_widget:
            display(self.sim.run_data)
        top_box = ipy.HBox(children=[self.run_slider, self.table_widget])
        self.step_slider = ipy.IntSlider(value=0, min=0, max=0, step=self.sim.microstructure_logging_interval,
                                         description='Step', orientation='Horizontal', layout=ipy.Layout(width='900px'))
        self.left_axis_selector = ipy.Dropdown(options=(), description='left (red)')
        self.right_axis_selector = ipy.Dropdown(options=(), description='right (blue)')
        selector_box = ipy.HBox(children=[self.left_axis_selector, self.right_axis_selector])
        plot_box = ipy.Output()
        with plot_box:
            self.fig.show()
        plot_box = ipy.VBox(children=[plot_box, selector_box])
        self.molecule_box = ipy.Output()
        self._redraw_molecule()
        bottom_box = ipy.HBox(children=[plot_box, self.molecule_box], layout=ipy.Layout(align_items='center'))
        self.display_widget = ipy.VBox(children=[top_box, self.step_slider, bottom_box])
        self._enable_observers()
    
    def display(self, sim) -> None:
        '''Creates an instance of the dashboard in the output of the notebook cell it is called in.'''
        self.sim = sim
        if self.sim.run_data['run'].max() > 0:
            self.run_slider.max = self.sim.run_data['run'].max()
            self.run_slider.min = 1
        display(self.display_widget)
    
    def live_update(self, sim, step: int, run_type: str, n_steps: int, temperature: float):
        '''Updates the dashboard live after new runs are started'''
        self.sim = sim
        if step > 0:
            self.step_slider.value = step
            self._redraw_molecule()
            self._redraw_plot()
            return
        
        #TODO freeze selectors during live update
        current_run_data = {'run': self.sim.run, 'type': run_type, 'n_steps': n_steps, 'T': temperature}
        with self.table_widget:
            clear_output()
            display(pd.concat([self.sim.run_data, pd.DataFrame(current_run_data, index=[0])]))
        self._disable_observers()
        self.run_slider.max += 1
        self.step_slider.max = ((n_steps-1) // self.sim.microstructure_logging_interval) * self.sim.microstructure_logging_interval
        self.run_slider.disabled = True
        self.step_slider.disabled = True
        self.run_slider.value = self.run_slider.max
        self._reset_axis_selectors()

    def reset(self, sim):
        '''Resets the dashboard at the end of a live display run so it can continue to be used'''
        self.sim = sim
        self.run_slider.min = 1
        self.run_slider.disabled = False
        self.step_slider.disabled = False
        with self.table_widget:
            clear_output()
            display(self.sim.run_data)
        self._enable_observers()
        self._redraw_plot()
        self.step_slider.value = self.step_slider.max

    def _step_slider_moved(self, _= None) -> None:  #placeholder arguement as widget observe calls method with redundant arguement 
        self.step_slider.value = self.step_slider.value
        self._redraw_molecule()
        self.line.set_xdata([self.step_slider.value])
        self.fig.canvas.draw()
    
    def _run_slider_moved(self, _= None) -> None:
        self.step_slider.value = 0
        self._reset_axis_selectors()
        self.step_slider.max = self.sim.step_data[self.run_slider.value].shape[0]-self.sim.microstructure_logging_interval
        self._redraw_plot()
        self._redraw_molecule()

    def _reset_axis_selectors(self):
        self._disable_observers()
        self.left_axis_selector.options = self.sim.step_data[self.run_slider.value].columns[1:]
        self.right_axis_selector.options = self.sim.step_data[self.run_slider.value].columns[1:]
        self.left_axis_selector.value, self.right_axis_selector.value = self.sim.step_data[self.run_slider.value].columns[-2:]
        self._enable_observers()

    def _disable_observers(self):
        if self.observers_enabled == False:
            return
        self.observers_enabled = False
        self.step_slider.unobserve(self._step_slider_moved, names='value')
        self.run_slider.unobserve(self._run_slider_moved, names='value')
        self.left_axis_selector.unobserve(self._redraw_plot, names='value')
        self.right_axis_selector.unobserve(self._redraw_plot, names='value')

    def _enable_observers(self):
        if self.observers_enabled:
            return
        self.observers_enabled = True
        self.step_slider.observe(self._step_slider_moved, names='value')
        self.run_slider.observe(self._run_slider_moved, names='value')
        self.left_axis_selector.observe(self._redraw_plot, names='value')
        self.right_axis_selector.observe(self._redraw_plot, names='value')

    def _redraw_molecule(self, _=None) -> None: 
        structure = self.sim.microstructures[self.run_slider.value][self.step_slider.value].copy()
        structure.insert(loc=0, column='element', value='C')
        molecule_xyz = f'{len(structure)}\n some comment\n {structure.to_string(header=False, index=False)}'
        self.mol_viewer.clear()
        self.mol_viewer.addModel(molecule_xyz, 'xyz')
        self.mol_viewer.setStyle({'stick': {'colorscheme': 'default'}, 'sphere': {'scale': 0.3, 'colorscheme': 'cyanCarbon'}})
        self.mol_viewer.zoomTo()
        with self.molecule_box:
            self.mol_viewer.update()
        
    def _redraw_plot(self, _=None) -> None: 
        left_data = self.sim.step_data[self.run_slider.value][self.left_axis_selector.value]
        right_data = self.sim.step_data[self.run_slider.value][self.right_axis_selector.value]
        self.left_ax.clear()
        self.right_ax.clear()
        self.left_ax.set_yscale('log' if left_data.max() > left_data.min()*100 and left_data.min() > 0 else 'linear')
        self.right_ax.set_yscale('log' if right_data.max() > right_data.min()*100 and right_data.min() > 0 else 'linear')
        self.right_ax.ticklabel_format(axis='x', style='sci', scilimits=(0,4))
        interval = 10 ** max(1, np.log10(len(left_data)).astype(int) - 3) #makes longer plots a bit faster
        self.left_ax.plot(np.arange(0, len(left_data), interval), left_data[::interval], color='red')
        self.right_ax.plot(np.arange(0, len(right_data), interval), right_data[::interval], color='blue')
        self.line = self.right_ax.axvline(x=self.step_slider.value, color='black', linestyle='--')
        self.fig.canvas.draw()