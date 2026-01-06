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

    def __init__(self, sim, show_config_panel: bool) -> None:
        '''Initiates internal methods, attributes, and dashboard widgets'''
        matplotlib.use('module://ipympl.backend_nbagg')
        self.sim = sim

        #top box
        self.table_widget = ipy.Output(layout=ipy.Layout(height='200px', overflow='auto'))
        with self.table_widget:
            display(self.sim.run_data)
        self.run_slider = ipy.IntSlider(value=0, min=0, max=0, description='Run', orientation='vertical')
        top_box = ipy.HBox(children=[self.run_slider, self.table_widget])

        self.step_slider = ipy.IntSlider(value=0, min=0, max=0, step=self.sim.microstructure_logging_interval,
                                         description='Step', orientation='Horizontal', layout=ipy.Layout(width='900px'))
        
        #data plotting
        with plt.ioff():
            self.fig, self.left_ax = plt.subplots(figsize=(6, 3))
            self.fig.tight_layout(rect=(0.05, 0.0, 0.95, 1.0))
            self.right_ax = self.left_ax.twinx()
            self.line = self.right_ax.axvline(x=0.5, color='black', linestyle='--')
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.footer_visible = False

        #plot box
        self.left_axis_selector = ipy.Dropdown(options=(), description='left (red)')
        self.right_axis_selector = ipy.Dropdown(options=(), description='right (blue)')
        selector_box = ipy.HBox(children=[self.left_axis_selector, self.right_axis_selector])
        figure_box = ipy.Output()
        with figure_box:
            self.fig.show()
        plot_box = ipy.VBox(children=[figure_box, selector_box])

        #molecule viewer
        self.mol_viewer = py3Dmol.view(width=300, height=300)
        self.mol_viewer_box = ipy.Output()

        #molecule box
        self.atom_radius_slider = ipy.FloatSlider(value=0.5, min=0.0, max=1.0, step = 0.01, orientation='Horizontal',
                                                   description='atom radius')
        self.atom_radius_slider.observe(self._redraw_molecule, names='value')
        self.bond_radius_slider = ipy.FloatSlider(value=0.2, min=0.0, max=1.0, step = 0.01, orientation='Horizontal',
                                                   description='bond radius')
        self.bond_radius_slider.observe(self._redraw_molecule, names='value')
        self.atom_colour_selector = ipy.Dropdown(options=colours, description='atom colour', value='cyan')
        self.atom_colour_selector.observe(self._redraw_molecule, names='value')
        self.bond_colour_selector = ipy.Dropdown(options=colours, description='bond colour', value='silver')
        self.bond_colour_selector.observe(self._redraw_molecule, names='value')
        mol_config_box = ipy.VBox([self.atom_radius_slider, self.bond_radius_slider,
                                        self.atom_colour_selector, self.bond_colour_selector])
        mol_display_options = ipy.Accordion([mol_config_box], titles=['molecule display options'])
        molecule_box = ipy.VBox([self.mol_viewer_box])
        if show_config_panel:
            molecule_box.children = [self.mol_viewer_box, mol_display_options]
        # self._redraw_molecule() #! temporarily disabled as causes molecule display to break
        
        #full display widget box
        bottom_box = ipy.HBox(children=[plot_box, molecule_box], layout=ipy.Layout(align_items='center'))
        self.display_widget = ipy.VBox(children=[top_box, self.step_slider, bottom_box])
        self.observers_enabled = False
        self._enable_observers()
    
    def display(self, sim) -> None:
        '''Creates an instance of the dashboard in the output of the notebook cell it is called in.'''
        self.sim = sim
        self._disable_observers() #! temporarily disabled as causes molecule display to break
        if self.sim.run_data['run'].max() > 0:
            self.run_slider.max = self.sim.run_data['run'].max()
            self.run_slider.min = 1
        self._enable_observers() #! temporarily as causes molecule display to break
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
            display(pd.concat([self.sim.run_data, pd.DataFrame(current_run_data, index=[0])]).reset_index(drop=True))
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

    def _step_slider_moved(self, _= None) -> None:  #placeholder arguement for widget observe calls method 
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
        bond_radius = 1e-5 if self.bond_radius_slider.value==0 else self.bond_radius_slider.value #allows hiding bond
        self.mol_viewer.setStyle({
            'stick': {'radius': bond_radius, 'color': self.bond_colour_selector.value},
            'sphere': {'radius': self.atom_radius_slider.value, 'color': self.atom_colour_selector.value}
            })
        self.mol_viewer.zoomTo()
        with self.mol_viewer_box: #moving this line to __init__ will always stop intial molecule from displaying?
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

colours = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
    'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
    'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
    'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
    'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
    'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream',
    'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid',
    'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
    'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
    'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen',
    'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
    'yellowgreen'
    ]
