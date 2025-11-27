#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive EoS Fitting GUI with Real-time Parameter Adjustment

Similar to EosFit7-GUI, allows manual parameter adjustment with live preview.

@author: candicewang928@gmail.com
Created: 2025-11-24
"""

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from crysfml_eos_module import CrysFMLEoS, EoSType, EoSParameters


class InteractiveEoSGUI:
    """
    Interactive GUI for EoS fitting with real-time parameter adjustment

    Features:
    - Manual parameter input (V0, B0, B0')
    - Parameter lock/unlock (fix or fit)
    - Real-time curve update
    - Automatic fitting
    - Quality metrics display
    - Data loading from CSV
    """

    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Interactive EoS Fitting - CrysFML Method")
        self.root.geometry("1480x900")

        # Palette for a calmer, consistent layout
        self.palette = {
            'background': '#f4f6fb',
            'panel_bg': '#ffffff',
            'section_header': '#dfe4ef',
            'accent': '#3f51b5',
            'text_primary': '#1f2933',
            'muted': '#5f6c7b',
        }

        self.root.configure(bg=self.palette['background'])

        # Data storage
        self.V_data = None
        self.P_data = None
        self.current_params = None
        self.fitted_params = None

        # EoS fitter
        self.eos_type = EoSType.BIRCH_MURNAGHAN_3RD
        self.fitter = None
        self.last_initial_params = None

        # Results window state
        self.results_window = None
        self.results_window_text = None
        self.last_results_output = ""

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container with adjustable left/right split
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                   sashrelief=tk.RAISED, sashwidth=8,
                                   opaqueresize=False, bg=self.palette['background'])
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = tk.Frame(main_pane, width=380, bg=self.palette['panel_bg'], bd=1, relief=tk.GROOVE)
        left_panel.pack_propagate(False)
        main_pane.add(left_panel, minsize=360)

        # Right panel - Plot and Results with adjustable height splitter
        right_panel = tk.PanedWindow(main_pane, orient=tk.VERTICAL,
                                     sashrelief=tk.RAISED, sashwidth=8,
                                     opaqueresize=False, bg=self.palette['background'])
        main_pane.add(right_panel)

        plot_container = tk.Frame(right_panel, bg=self.palette['background'])
        results_container = tk.Frame(right_panel, bg=self.palette['background'])

        right_panel.add(plot_container, minsize=360)
        right_panel.add(results_container, minsize=180)

        # Setup left panel sections
        self.setup_data_section(left_panel)
        self.setup_eos_selection(left_panel)
        self.setup_parameters_section(left_panel)
        self.setup_fitting_section(left_panel)

        # Setup right panel
        self.setup_plot(plot_container)
        self.setup_results_section(results_container)

    def setup_data_section(self, parent):
        """Setup data loading section"""
        frame = tk.LabelFrame(parent, text="  Data  ", bg=self.palette['panel_bg'],
                             fg=self.palette['text_primary'], font=('Arial', 10, 'bold'),
                             relief=tk.FLAT, bd=0, padx=4, pady=2, labelanchor='nw')
        frame.pack(fill=tk.X, pady=(0, 8))

        inner_frame = tk.Frame(frame, bg=self.palette['panel_bg'], padx=8, pady=8)
        inner_frame.pack(fill=tk.X)

        btn_style = {'font': ('Arial', 10, 'bold'), 'width': 24, 'height': 2,
                     'bg': self.palette['accent'], 'fg': 'white', 'relief': tk.FLAT, 'bd': 0}

        tk.Button(inner_frame, text="Load CSV File", command=self.load_csv,
                 **btn_style).pack(fill=tk.X, pady=3)

        # Data info
        self.data_info_label = tk.Label(inner_frame, text="No data loaded",
                                        font=('Arial', 9), bg=self.palette['panel_bg'], fg=self.palette['muted'],
                                        wraplength=320, justify=tk.LEFT)
        self.data_info_label.pack(fill=tk.X, pady=(8, 0))

    def setup_eos_selection(self, parent):
        """Setup EoS model selection"""
        frame = tk.LabelFrame(parent, text="  EoS Model  ", bg=self.palette['panel_bg'],
                             fg=self.palette['text_primary'], font=('Arial', 10, 'bold'),
                             relief=tk.FLAT, bd=0, padx=4, pady=2, labelanchor='nw')
        frame.pack(fill=tk.X, pady=(0, 8))

        inner_frame = tk.Frame(frame, bg=self.palette['panel_bg'], padx=8, pady=8)
        inner_frame.pack(fill=tk.X)

        self.eos_var = tk.StringVar(value="Birch-Murnaghan 3rd")

        eos_options = [
            "Birch-Murnaghan 2nd",
            "Birch-Murnaghan 3rd",
            "Birch-Murnaghan 4th",
            "Murnaghan",
            "Vinet",
            "Natural Strain"
        ]

        tk.Label(inner_frame, text="Model:", bg=self.palette['panel_bg'], fg=self.palette['text_primary'], font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        eos_combo = ttk.Combobox(inner_frame, textvariable=self.eos_var,
                                 values=eos_options, state='readonly', width=23, font=('Arial', 9))
        eos_combo.pack(fill=tk.X, pady=5)
        eos_combo.bind('<<ComboboxSelected>>', self.on_eos_changed)

    def setup_parameters_section(self, parent):
        """Setup parameters input section"""
        frame = tk.LabelFrame(parent, text="  EoS Parameters  ", bg=self.palette['panel_bg'],
                             fg=self.palette['text_primary'], font=('Arial', 10, 'bold'),
                             relief=tk.FLAT, bd=0, padx=4, pady=2, labelanchor='nw')
        frame.pack(fill=tk.X, pady=(0, 8))

        inner_frame = tk.Frame(frame, bg=self.palette['panel_bg'], padx=8, pady=8)
        inner_frame.pack(fill=tk.X)

        # Create parameter entries
        self.param_vars = {}
        self.param_lock_vars = {}
        self.param_entries = {}

        params = [
            ('V0', 'V₀ (Å³/atom)', 11.5),
            ('B0', 'B₀ (GPa)', 130.0),
            ('B0_prime', "B₀'", 4.0),
        ]

        # Header
        tk.Label(inner_frame, text="Parameter", font=('Arial', 9, 'bold'),
                bg=self.palette['panel_bg'], fg=self.palette['text_primary']).grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        tk.Label(inner_frame, text="Value", font=('Arial', 9, 'bold'),
                bg=self.palette['panel_bg'], fg=self.palette['text_primary']).grid(row=0, column=1, padx=5, pady=3)
        tk.Label(inner_frame, text="Lock", font=('Arial', 9, 'bold'),
                bg=self.palette['panel_bg'], fg=self.palette['text_primary']).grid(row=0, column=2, padx=5, pady=3)

        for idx, (key, label, default) in enumerate(params, start=1):
            # Parameter label
            tk.Label(inner_frame, text=label, bg=self.palette['panel_bg'], fg=self.palette['text_primary'], font=('Arial', 9)).grid(
                row=idx, column=0, padx=5, pady=4, sticky=tk.W)

            # Value entry
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            entry = tk.Entry(inner_frame, textvariable=var, width=14,
                           font=('Arial', 9), bg='white', fg=self.palette['text_primary'])
            entry.grid(row=idx, column=1, padx=5, pady=4)
            entry.bind('<Return>', lambda e: self.update_manual_fit())
            entry.bind('<FocusOut>', lambda e: self.update_manual_fit())
            self.param_entries[key] = entry

            # Lock checkbox
            lock_var = tk.BooleanVar(value=False)
            self.param_lock_vars[key] = lock_var
            tk.Checkbutton(inner_frame, variable=lock_var, bg=self.palette['panel_bg']).grid(
                row=idx, column=2, padx=5, pady=4)

        # Real-time update button
        tk.Button(inner_frame, text="Update Plot", command=self.update_manual_fit,
                 font=('Arial', 9, 'bold'), bg='#e6ecfb', fg=self.palette['text_primary'],
                 relief=tk.FLAT, bd=0, activebackground='#d4ddf5').grid(
            row=len(params)+1, column=0, columnspan=3, padx=5, pady=(10, 0), sticky=tk.EW)

    def setup_fitting_section(self, parent):
        """Setup fitting control section"""
        frame = tk.LabelFrame(parent, text="  Fitting Control  ", bg=self.palette['panel_bg'],
                             fg=self.palette['text_primary'], font=('Arial', 10, 'bold'),
                             relief=tk.FLAT, bd=0, padx=4, pady=2, labelanchor='nw')
        frame.pack(fill=tk.X, pady=(0, 8))

        inner_frame = tk.Frame(frame, bg=self.palette['panel_bg'], padx=8, pady=8)
        inner_frame.pack(fill=tk.X)

        # Regularization strength control (for B0' constraint)
        reg_frame = tk.Frame(inner_frame, bg=self.palette['panel_bg'])
        reg_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(reg_frame, text="B0' Regularization:", bg=self.palette['panel_bg'],
                fg=self.palette['text_primary'], font=('Arial', 9, 'bold')).pack(anchor=tk.W)

        self.reg_strength = tk.DoubleVar(value=1.0)
        reg_slider_frame = tk.Frame(reg_frame, bg=self.palette['panel_bg'])
        reg_slider_frame.pack(fill=tk.X, pady=2)

        reg_slider = tk.Scale(reg_slider_frame, from_=0.1, to=10.0, resolution=0.1,
                             orient=tk.HORIZONTAL, variable=self.reg_strength,
                             bg=self.palette['panel_bg'], font=('Arial', 8), length=180,
                             troughcolor='#d9dfee', highlightthickness=0)
        reg_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.reg_label = tk.Label(reg_slider_frame, text=f"{self.reg_strength.get():.1f}",
                                 bg=self.palette['panel_bg'], fg=self.palette['text_primary'], font=('Arial', 9), width=4)
        self.reg_label.pack(side=tk.LEFT, padx=(5, 0))

        def update_reg_label(*args):
            self.reg_label.config(text=f"{self.reg_strength.get():.1f}")

        self.reg_strength.trace('w', update_reg_label)

        tk.Label(reg_frame, text="(Higher = stronger constraint to B0'=4)",
                bg=self.palette['panel_bg'], font=('Arial', 8, 'italic'), fg=self.palette['muted']).pack(anchor=tk.W)

        # Fitting buttons
        btn_style = {'font': ('Arial', 9, 'bold'), 'width': 25, 'height': 1,
                     'relief': tk.FLAT, 'bd': 0, 'activebackground': '#e1e7f5',
                     'fg': self.palette['text_primary']}

        tk.Button(inner_frame, text="Fit Unlocked Parameters",
                 command=self.fit_unlocked, bg='#dbe3f9',
                 **btn_style).pack(fill=tk.X, pady=3)

        tk.Button(inner_frame, text="Try Multiple Strategies",
                 command=self.fit_multiple_strategies, bg='#e9eefc',
                 **btn_style).pack(fill=tk.X, pady=3)

        tk.Button(inner_frame, text="Reset to Initial Guess",
                 command=self.reset_parameters, bg='#f1f4fb',
                 **btn_style).pack(fill=tk.X, pady=3)


    def setup_plot(self, parent):
        """Setup matplotlib plot"""
        # Plot frame
        plot_frame = tk.Frame(parent, bg=self.palette['panel_bg'], relief=tk.FLAT, bd=1,
                              highlightthickness=1, highlightbackground='#e0e6f5')
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8), padx=(0, 4))

        # Create figure with single subplot
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self.ax_main = self.fig.add_subplot(111)

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        self.fig.tight_layout()

    def setup_results_section(self, parent):
        """Setup results display section (similar to EosFit7 output)"""
        frame = tk.LabelFrame(parent, text="  Fitting Results  ", bg=self.palette['panel_bg'],
                             fg=self.palette['text_primary'], font=('Arial', 10, 'bold'),
                             relief=tk.FLAT, bd=0, padx=6, pady=4, labelanchor='nw')
        frame.pack(fill=tk.BOTH, expand=True)

        inner_frame = tk.Frame(frame, bg=self.palette['panel_bg'], padx=8, pady=8)
        inner_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(inner_frame, text="The preview matches the floating information window.",
                 bg=self.palette['panel_bg'], fg=self.palette['muted'], font=('Arial', 9, 'italic')).pack(anchor=tk.W)

        tk.Button(inner_frame, text="Open Fitting Information Window",
                  command=self.open_results_window,
                  font=('Arial', 9, 'bold'), bg='#e6ecfb', fg=self.palette['text_primary'],
                  relief=tk.FLAT, bd=0, activebackground='#d4ddf5').pack(fill=tk.X, pady=(6, 10))

        self.preview_text = tk.Text(inner_frame, height=8, width=80,
                                    font=('Courier New', 9), wrap=tk.NONE,
                                    bg='#f9fbff', fg=self.palette['text_primary'], state='disabled',
                                    relief=tk.FLAT, bd=1, highlightthickness=1,
                                    highlightbackground='#e0e6f5')
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        self.update_results_display()

    def load_csv(self):
        """Load data from CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            df = pd.read_csv(filename)

            # Check required columns
            if 'V_atomic' not in df.columns or 'Pressure (GPa)' not in df.columns:
                messagebox.showerror("Error",
                    "CSV must contain 'V_atomic' and 'Pressure (GPa)' columns")
                return

            self.V_data = df['V_atomic'].dropna().values
            self.P_data = df['Pressure (GPa)'].dropna().values

            # Ensure same length
            min_len = min(len(self.V_data), len(self.P_data))
            self.V_data = self.V_data[:min_len]
            self.P_data = self.P_data[:min_len]

            # Update GUI components
            self.update_data_info()
            self.reset_parameters()
            self.update_plot()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")

    def update_data_info(self):
        """Update data information display"""
        if self.V_data is None or self.P_data is None:
            self.data_info_label.config(text="No data loaded")
            return

        info = f"Data points: {len(self.V_data)}\n"
        info += f"V: {self.V_data.min():.3f} - {self.V_data.max():.3f} Å³\n"
        info += f"P: {self.P_data.min():.2f} - {self.P_data.max():.2f} GPa"
        self.data_info_label.config(text=info)

    def on_eos_changed(self, event=None):
        """Handle EoS model change - only update fitter, keep current parameters"""
        eos_map = {
            "Birch-Murnaghan 2nd": EoSType.BIRCH_MURNAGHAN_2ND,
            "Birch-Murnaghan 3rd": EoSType.BIRCH_MURNAGHAN_3RD,
            "Birch-Murnaghan 4th": EoSType.BIRCH_MURNAGHAN_4TH,
            "Murnaghan": EoSType.MURNAGHAN,
            "Vinet": EoSType.VINET,
            "Natural Strain": EoSType.NATURAL_STRAIN
        }

        self.eos_type = eos_map.get(self.eos_var.get(), EoSType.BIRCH_MURNAGHAN_3RD)
        self.fitter = CrysFMLEoS(eos_type=self.eos_type,
                                         regularization_strength=self.reg_strength.get())
        # Don't reset parameters when switching models - keep current values
        # Only update the plot with the new EoS type
        if self.V_data is not None and self.P_data is not None:
            self.update_manual_fit()

    def reset_parameters(self):
        """Reset parameters to smart initial guess"""
        if self.V_data is None or self.P_data is None:
            return

        self.fitter = CrysFMLEoS(eos_type=self.eos_type,
                                         regularization_strength=self.reg_strength.get())

        # Get smart initial guess
        if hasattr(self.fitter, '_smart_initial_guess'):
            V0_guess, B0_guess, B0_prime_guess = self.fitter._smart_initial_guess(
                self.V_data, self.P_data
            )
        else:
            # Fallback to simple guess
            V0_guess = self.V_data.max() * 1.05
            B0_guess = 130.0
            B0_prime_guess = 4.0

        self.param_vars['V0'].set(round(V0_guess, 4))
        self.param_vars['B0'].set(round(B0_guess, 2))
        self.param_vars['B0_prime'].set(round(B0_prime_guess, 3))

        self.update_manual_fit()

    def get_current_params(self):
        """Get current parameter values from GUI"""
        params = EoSParameters(eos_type=self.eos_type)
        params.V0 = self.param_vars['V0'].get()
        params.B0 = self.param_vars['B0'].get()
        params.B0_prime = self.param_vars['B0_prime'].get()

        return params

    def update_manual_fit(self):
        """Update plot with current manual parameters"""
        if self.V_data is None or self.P_data is None:
            return

        # Ensure the fitter uses the current EoS selection and the
        # CrysFML-style pressure evaluation implemented in
        # ``crysfml_eos_module.py``
        if self.fitter is None or self.fitter.eos_type != self.eos_type:
            self.fitter = CrysFMLEoS(
                eos_type=self.eos_type,
                regularization_strength=self.reg_strength.get(),
            )

        params = self.get_current_params()

        # Calculate fitted pressures
        try:
            P_fit = self.fitter.calculate_pressure(self.V_data, params)

            # Calculate statistics
            residuals = self.P_data - P_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.P_data - np.mean(self.P_data))**2)
            params.R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            params.RMSE = np.sqrt(np.mean(residuals**2))

            self.current_params = params
            self.update_plot()

        except Exception as e:
            print(f"Error calculating pressure: {e}")

    def auto_fit_all(self):
        """Perform automatic fitting of all parameters"""
        if self.V_data is None or self.P_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        self.fitter = CrysFMLEoS(eos_type=self.eos_type,
                                         regularization_strength=self.reg_strength.get())

        # Try fitting with smart guess first
        self.last_initial_params = self.get_current_params()
        params = self.fitter.fit(self.V_data, self.P_data, use_smart_guess=True)

        # If initial fit fails, try multiple strategies
        if params is None:
            print("Initial fit failed, trying multiple strategies...")
            params = self.fitter.fit_with_multiple_strategies(
                self.V_data, self.P_data, verbose=True
            )

        # If still no params, use current manual values as fallback
        if params is None:
            print("All automatic strategies failed, using current manual parameters...")
            params = self.get_current_params()
            
            # Calculate statistics for current parameters
            try:
                P_fit = self.fitter.calculate_pressure(self.V_data, params)
                residuals = self.P_data - P_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((self.P_data - np.mean(self.P_data))**2)
                params.R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                params.RMSE = np.sqrt(np.mean(residuals**2))
            except:
                pass

        # Always update GUI with best available parameters
        self.param_vars['V0'].set(round(params.V0, 4))
        self.param_vars['B0'].set(round(params.B0, 2))
        self.param_vars['B0_prime'].set(round(params.B0_prime, 3))

        self.fitted_params = params
        self.current_params = params
        self.update_plot()

    def fit_unlocked(self):
        """Fit only unlocked parameters"""
        if self.V_data is None or self.P_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        # Get locked status
        V0_locked = self.param_lock_vars['V0'].get()
        B0_locked = self.param_lock_vars['B0'].get()
        B0_prime_locked = self.param_lock_vars['B0_prime'].get()

        # If all locked, nothing to fit
        if V0_locked and B0_locked and B0_prime_locked:
            messagebox.showwarning("Warning", "All parameters are locked!")
            return

        # Use the CrysFML-style fitter with lock-aware bounds
        self.fitter = CrysFMLEoS(eos_type=self.eos_type,
                                         regularization_strength=self.reg_strength.get())

        try:
            self.last_initial_params = self.get_current_params()
            lock_flags = {
                'V0': V0_locked,
                'B0': B0_locked,
                'B0_prime': B0_prime_locked,
            }

            params = self.fitter.fit(
                self.V_data,
                self.P_data,
                use_smart_guess=True,
                initial_params=self.get_current_params(),
                lock_flags=lock_flags,
            )

            if params is not None:
                if not V0_locked:
                    self.param_vars['V0'].set(round(params.V0, 4))
                if not B0_locked:
                    self.param_vars['B0'].set(round(params.B0, 2))
                if not B0_prime_locked:
                    self.param_vars['B0_prime'].set(round(params.B0_prime, 3))

                self.fitted_params = params
                self.current_params = params
                self.update_plot()
            else:
                self.update_manual_fit()

        except Exception as e:
            print(f"Fit unlocked error: {e}")
            self.update_manual_fit()

    def fit_multiple_strategies(self):
        """Try fitting with multiple strategies"""
        if self.V_data is None or self.P_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        self.fitter = CrysFMLEoS(eos_type=self.eos_type,
                                         regularization_strength=self.reg_strength.get())

        try:
            self.last_initial_params = self.get_current_params()
            # Print to console for user to see progress
            print("\n" + "="*60)
            print("Trying multiple fitting strategies...")
            print("="*60)
            
            params = self.fitter.fit_with_multiple_strategies(
                self.V_data, self.P_data, verbose=True
            )

            if params is not None:
                self.param_vars['V0'].set(round(params.V0, 4))
                self.param_vars['B0'].set(round(params.B0, 2))
                self.param_vars['B0_prime'].set(round(params.B0_prime, 3))

                self.fitted_params = params
                self.current_params = params
                self.update_plot()

                print("="*60)
                print("Best fit found!")
                print("="*60 + "\n")
            else:
                print("="*60)
                print("All strategies failed - using current manual values")
                print("="*60 + "\n")
                self.update_manual_fit()

        except Exception as e:
            print(f"Error in multiple strategies: {e}")
            self.update_manual_fit()

    def update_plot(self):
        """Update the plot with current data and fit"""
        if self.V_data is None or self.P_data is None:
            return

        # Clear axis
        self.ax_main.clear()

        # Main P-V plot
        self.ax_main.scatter(self.V_data, self.P_data, s=60, c='#2196F3',
                           marker='o', label='Experimental Data',
                           alpha=0.8, edgecolors='#0D47A1', linewidths=1.5, zorder=5)

        # Plot current fit if available
        if self.current_params is not None:
            V_fit = np.linspace(self.V_data.min()*0.95, self.V_data.max()*1.05, 300)
            P_fit = self.fitter.calculate_pressure(V_fit, self.current_params)

            self.ax_main.plot(V_fit, P_fit, 'r-', linewidth=2.5,
                            label=f'Fitted Curve (R²={self.current_params.R_squared:.4f})',
                            alpha=0.9, zorder=3)

        self.ax_main.set_xlabel('Volume V (Å³/atom)', fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel('Pressure P (GPa)', fontsize=12, fontweight='bold')
        self.ax_main.set_title(f'{self.eos_type.value.replace("_", " ").title()} Equation of State',
                              fontsize=13, fontweight='bold')
        self.ax_main.legend(loc='best', fontsize=11, framealpha=0.9)
        self.ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Auto-scale axes to fit data
        self.ax_main.autoscale(enable=True, axis='both', tight=False)

        self.fig.tight_layout()
        self.canvas.draw()

        # Update results display
        self.update_results_display()

    def update_results_display(self):
        """Update results display (EosFit7-style summary preview)"""
        if self.preview_text is None:
            return

        self.preview_text.configure(state='normal')
        self.preview_text.delete(1.0, tk.END)

        text = self._format_results_output()

        self.preview_text.insert(tk.END, text)
        self.preview_text.configure(state='disabled')
        self.last_results_output = text
        self._refresh_results_window()

    def _format_results_output(self):
        """Create a compact, CrysFML-style summary for preview and popup."""
        if self.current_params is None or self.V_data is None or self.P_data is None:
            return "No fitting results yet.\n\nLoad data and adjust parameters or run a fit."

        if self.fitter is None:
            return "No fitter available for displaying results."

        params = self.current_params

        try:
            P_fit = self.fitter.calculate_pressure(self.V_data, params)
            residuals = self.P_data - P_fit
        except Exception:
            P_fit, residuals = None, None

        cycles = []
        cycles.append(self._format_cycle_output("RESULTS FROM CYCLE 1", params,
                                                self.last_initial_params, P_fit, residuals))

        if self.last_initial_params is not None and self.last_initial_params is not params:
            try:
                start_P_fit = self.fitter.calculate_pressure(self.V_data, self.last_initial_params)
                start_residuals = self.P_data - start_P_fit
            except Exception:
                start_P_fit, start_residuals = None, None

            cycles.append(self._format_cycle_output("RESULTS FROM START", self.last_initial_params,
                                                    None, start_P_fit, start_residuals))

        return "\n\n".join(cycles)

    def _format_cycle_output(self, title, params, reference_params, P_fit, residuals):
        lines = [title, "=" * 72, ""]
        lines.append("PARA  REF          NEW        SHIFT       E.S.D.     SHIFT/ERROR")
        lines.append("-" * 72)

        lock_flags = {
            'V0': self.param_lock_vars.get('V0', tk.BooleanVar(value=False)).get(),
            'B0': self.param_lock_vars.get('B0', tk.BooleanVar(value=False)).get(),
            'B0_prime': self.param_lock_vars.get('B0_prime', tk.BooleanVar(value=False)).get(),
        }

        def fmt_param(label, value, err, ref_key):
            ref_locked = lock_flags.get(ref_key, False)
            ref_marker = 0 if ref_locked else 1
            ref_value = reference_params.__dict__.get(ref_key) if reference_params is not None else value
            shift = value - ref_value if ref_value is not None else 0.0
            esd = err if err is not None else 0.0
            shift_over_err = (shift / esd) if esd not in (0, None) else 0.0
            return f"{label:<4}{ref_marker:>2}   {value:10.5f}   {shift:10.5f}   {esd:10.5f}   {shift_over_err:8.2f}"

        lines.append(fmt_param('V0', params.V0, getattr(params, 'V0_err', 0.0), 'V0'))
        lines.append(fmt_param('K0', params.B0, getattr(params, 'B0_err', 0.0), 'B0'))

        kp_locked = lock_flags.get('B0_prime', False)
        kp_marker = 0 if kp_locked else 1
        kp_shift = params.B0_prime - (reference_params.B0_prime if (reference_params and reference_params.B0_prime is not None) else params.B0_prime)
        kp_line = f"Kp   {kp_marker:>1}   {params.B0_prime:10.5f}"
        if kp_locked:
            kp_line += "   [NOT REFINED]"
        else:
            kp_esd = getattr(params, 'B0_prime_err', 0.0)
            kp_shift_over_err = (kp_shift / kp_esd) if kp_esd not in (0, None) else 0.0
            kp_line += f"   {kp_shift:10.5f}   {kp_esd:10.5f}   {kp_shift_over_err:8.2f}"
        lines.append(kp_line)

        kpp_val = getattr(params, 'B0_prime2', 0.0)
        lines.append(f"Kpp  0   {kpp_val:10.5f}   [IMPLIED VALUE]")

        if residuals is not None and len(residuals) > 0:
            chi_value = params.chi2 if getattr(params, 'chi2', 0) else 1.00
            max_idx = int(np.argmax(np.abs(residuals)))
            max_residual = residuals[max_idx]
        else:
            chi_value = params.chi2 if getattr(params, 'chi2', 0) else 1.00
            max_residual = 0.0

        lines.append("")
        lines.append(f"W-CHI^2 = {chi_value:5.2f} (AND ESD'S RESCALED BY W-CHI^2)")
        lines.append(f"MAXIMUM DELTA-PRESSURE = {max_residual:+.2f}")

        return "\n".join(lines)

    def open_results_window(self):
        """Open a toplevel window that mirrors the fitting output."""
        if self.results_window is not None and tk.Toplevel.winfo_exists(self.results_window):
            self.results_window.lift()
            self._refresh_results_window()
            return

        self.results_window = tk.Toplevel(self.root)
        self.results_window.title("Fitting Information Window")
        self.results_window.geometry("820x480")
        self.results_window.configure(bg=self.palette['panel_bg'])

        container = tk.Frame(self.results_window, bg=self.palette['panel_bg'], padx=10, pady=10)
        container.pack(fill=tk.BOTH, expand=True)

        self.results_window_text = tk.Text(container, height=18, width=90,
                                          font=('Courier New', 10), wrap=tk.NONE,
                                          bg='#f9fbff', fg=self.palette['text_primary'],
                                          relief=tk.FLAT, bd=1, highlightthickness=1,
                                          highlightbackground='#e0e6f5')
        self.results_window_text.pack(fill=tk.BOTH, expand=True)

        v_scroll = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.results_window_text.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_window_text.configure(yscrollcommand=v_scroll.set)

        h_scroll = tk.Scrollbar(container, orient=tk.HORIZONTAL, command=self.results_window_text.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_window_text.configure(xscrollcommand=h_scroll.set)

        self.results_window.protocol("WM_DELETE_WINDOW", self._close_results_window)
        self._refresh_results_window()

    def _refresh_results_window(self):
        """Push the latest text into the floating results window."""
        if (self.results_window is None or
                self.results_window_text is None or
                not tk.Toplevel.winfo_exists(self.results_window)):
            return

        self.results_window_text.delete(1.0, tk.END)
        self.results_window_text.insert(tk.END, self.last_results_output)

    def _close_results_window(self):
        """Reset references when the floating window is closed."""
        if self.results_window is not None:
            self.results_window.destroy()
        self.results_window = None
        self.results_window_text = None


def main():
    """Main function to run the GUI"""
    root = tk.Tk()

    # Configure ttk style
    style = ttk.Style()
    style.theme_use('clam')  # or 'alt', 'default', 'classic'

    # Configure custom button style
    style.configure('Accent.TButton', font=('Arial', 9, 'bold'))

    app = InteractiveEoSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()