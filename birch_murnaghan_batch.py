
# -*- coding: utf-8 -*-
"""
Birch-Murnaghan Equation Fitting for Pressure-Volume Curves - Class Implementation
For fitting pressure-volume data and calculating bulk modulus parameters
@author: candicewang928@gmail.com
Created: 2025-11-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Configure matplotlib to properly display special characters and symbols
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'


class BirchMurnaghanFitter:
    """
    Birch-Murnaghan Equation of State Fitter

    This class provides comprehensive fitting of pressure-volume data using
    2nd and 3rd order Birch-Murnaghan equations of state, with support for:
    - Automatic parameter estimation
    - Statistical analysis (R², RMSE)
    - Visualization of fitting results
    - Export of results to CSV
    """

    def __init__(self, V0_bounds=(0.8, 1.3), B0_bounds=(50, 500),
                 B0_prime_bounds=(2.5, 6.5), max_iterations=10000):
        """
        Initialize Birch-Murnaghan Fitter

        Parameters:
        -----------
        V0_bounds : tuple
            Bounds for V0 as multipliers of max experimental volume (default: (0.8, 1.3))
        B0_bounds : tuple
            Bounds for bulk modulus B0 in GPa (default: (50, 500))
        B0_prime_bounds : tuple
            Bounds for B0' (default: (2.5, 6.5))
        max_iterations : int
            Maximum iterations for curve fitting (default: 10000)
        """
        self.V0_bounds = V0_bounds
        self.B0_bounds = B0_bounds
        self.B0_prime_bounds = B0_prime_bounds
        self.max_iterations = max_iterations

        # Storage for results
        self.results_original = None
        self.results_new = None
        self.V_original = None
        self.P_original = None
        self.V_new = None
        self.P_new = None

    # ==================== Static Methods: Equations of State ====================

    @staticmethod
    def birch_murnaghan_2nd(V, V0, B0):
        """
        2nd order Birch-Murnaghan Equation of State

        Parameters:
        -----------
        V : float or array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)

        Returns:
        --------
        P : float or array
            Pressure (GPa)
        """
        eta = (V0 / V) ** (1/3)
        P = 3 * B0 / 2 * (eta**7 - eta**5)
        return P

    @staticmethod
    def birch_murnaghan_3rd(V, V0, B0, B0_prime):
        """
        3rd order Birch-Murnaghan Equation of State

        Parameters:
        -----------
        V : float or array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            First pressure derivative of bulk modulus (dimensionless)

        Returns:
        --------
        P : float or array
            Pressure (GPa)
        """
        eta = (V0 / V) ** (1/3)
        P = 3 * B0 / 2 * (eta**7 - eta**5) * (1 + 0.75 * (B0_prime - 4) * (eta**2 - 1))
        return P

    # ==================== Data Loading ====================

    def load_data_from_csv(self, original_csv, new_csv):
        """
        Load pressure-volume data from CSV files

        Parameters:
        -----------
        original_csv : str
            Path to CSV file containing original phase data
        new_csv : str
            Path to CSV file containing new phase data

        Returns:
        --------
        bool : True if successful, False otherwise
        """
        try:
            df_orig = pd.read_csv(original_csv)
            df_new = pd.read_csv(new_csv)

            # Check required columns
            required_columns = ['V_atomic', 'Pressure (GPa)']
            for col in required_columns:
                if col not in df_orig.columns or col not in df_new.columns:
                    print(f"❌ Error: Required column '{col}' missing in data files")
                    return False

            # Extract data and remove null values
            V_orig = df_orig['V_atomic'].dropna().values
            P_orig = df_orig['Pressure (GPa)'].dropna().values
            V_new = df_new['V_atomic'].dropna().values
            P_new = df_new['Pressure (GPa)'].dropna().values

            # Ensure data pairing
            min_len_orig = min(len(V_orig), len(P_orig))
            self.V_original = V_orig[:min_len_orig]
            self.P_original = P_orig[:min_len_orig]

            min_len_new = min(len(V_new), len(P_new))
            self.V_new = V_new[:min_len_new]
            self.P_new = P_new[:min_len_new]

            print(f"\n✅ Data loaded successfully!")
            print(f"   Original phase data points: {len(self.V_original)}")
            print(f"   New phase data points: {len(self.V_new)}")

            return True

        except FileNotFoundError as e:
            print(f"❌ Error: Data files not found - {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def set_data_manually(self, V_original, P_original, V_new, P_new):
        """
        Manually set pressure-volume data

        Parameters:
        -----------
        V_original : array
            Volume data for original phase (Å³/atom)
        P_original : array
            Pressure data for original phase (GPa)
        V_new : array
            Volume data for new phase (Å³/atom)
        P_new : array
            Pressure data for new phase (GPa)
        """
        self.V_original = np.array(V_original)
        self.P_original = np.array(P_original)
        self.V_new = np.array(V_new)
        self.P_new = np.array(P_new)

        print(f"\n✅ Data set manually!")
        print(f"   Original phase data points: {len(self.V_original)}")
        print(f"   New phase data points: {len(self.V_new)}")

    # ==================== Fitting Functions ====================

    def fit_single_phase(self, V_data, P_data, phase_name=""):
        """
        Fit 2nd and 3rd order Birch-Murnaghan equations to P-V data

        Parameters:
        -----------
        V_data : array
            Volume data array
        P_data : array
            Pressure data array
        phase_name : str
            Phase name for output display

        Returns:
        --------
        results : dict
            Dictionary containing fitting parameters and statistics
        """
        results = {}

        # Initial guess values
        V0_guess = np.max(V_data) * 1.02
        B0_guess = 150
        B0_prime_guess = 4.0

        # ==================== 2nd order BM equation fitting ====================
        bounds_2nd = ([np.max(V_data) * self.V0_bounds[0], self.B0_bounds[0]],
                      [np.max(V_data) * self.V0_bounds[1], self.B0_bounds[1]])

        try:
            popt_2nd, pcov_2nd = curve_fit(
                self.birch_murnaghan_2nd,
                V_data,
                P_data,
                p0=[V0_guess, B0_guess],
                bounds=bounds_2nd,
                maxfev=self.max_iterations
            )

            V0_2nd, B0_2nd = popt_2nd
            perr_2nd = np.sqrt(np.diag(pcov_2nd))

            # Calculate fitting residuals and R²
            P_fit_2nd = self.birch_murnaghan_2nd(V_data, *popt_2nd)
            residuals_2nd = P_data - P_fit_2nd
            ss_res_2nd = np.sum(residuals_2nd**2)
            ss_tot_2nd = np.sum((P_data - np.mean(P_data))**2)
            r_squared_2nd = 1 - (ss_res_2nd / ss_tot_2nd)
            rmse_2nd = np.sqrt(np.mean(residuals_2nd**2))

            results['2nd_order'] = {
                'V0': V0_2nd,
                'V0_err': perr_2nd[0],
                'B0': B0_2nd,
                'B0_err': perr_2nd[1],
                'B0_prime': 4.0,
                'B0_prime_err': 0,
                'R_squared': r_squared_2nd,
                'RMSE': rmse_2nd,
                'fitted_P': P_fit_2nd
            }

            print(f"\n{'='*60}")
            print(f"{phase_name} - 2nd Order Birch-Murnaghan Fitting Results:")
            print(f"{'='*60}")
            print(f"V₀ = {V0_2nd:.4f} ± {perr_2nd[0]:.4f} Å³/atom")
            print(f"B₀ = {B0_2nd:.2f} ± {perr_2nd[1]:.2f} GPa")
            print(f"B₀' = 4.0 (fixed)")
            print(f"R² = {r_squared_2nd:.6f}")
            print(f"RMSE = {rmse_2nd:.4f} GPa")

        except Exception as e:
            print(f"⚠️ {phase_name} - 2nd order BM fitting failed: {e}")
            results['2nd_order'] = None

        # ==================== 3rd order BM equation fitting ====================
        bounds_3rd = ([np.max(V_data) * self.V0_bounds[0], self.B0_bounds[0], self.B0_prime_bounds[0]],
                      [np.max(V_data) * self.V0_bounds[1], self.B0_bounds[1], self.B0_prime_bounds[1]])

        try:
            popt_3rd, pcov_3rd = curve_fit(
                self.birch_murnaghan_3rd,
                V_data,
                P_data,
                p0=[V0_guess, B0_guess, B0_prime_guess],
                bounds=bounds_3rd,
                maxfev=self.max_iterations
            )

            V0_3rd, B0_3rd, B0_prime_3rd = popt_3rd
            perr_3rd = np.sqrt(np.diag(pcov_3rd))

            # Calculate fitting residuals and R²
            P_fit_3rd = self.birch_murnaghan_3rd(V_data, *popt_3rd)
            residuals_3rd = P_data - P_fit_3rd
            ss_res_3rd = np.sum(residuals_3rd**2)
            ss_tot_3rd = np.sum((P_data - np.mean(P_data))**2)
            r_squared_3rd = 1 - (ss_res_3rd / ss_tot_3rd)
            rmse_3rd = np.sqrt(np.mean(residuals_3rd**2))

            results['3rd_order'] = {
                'V0': V0_3rd,
                'V0_err': perr_3rd[0],
                'B0': B0_3rd,
                'B0_err': perr_3rd[1],
                'B0_prime': B0_prime_3rd,
                'B0_prime_err': perr_3rd[2],
                'R_squared': r_squared_3rd,
                'RMSE': rmse_3rd,
                'fitted_P': P_fit_3rd
            }

            print(f"\n{'='*60}")
            print(f"{phase_name} - 3rd Order Birch-Murnaghan Fitting Results:")
            print(f"{'='*60}")
            print(f"V₀ = {V0_3rd:.4f} ± {perr_3rd[0]:.4f} Å³/atom")
            print(f"B₀ = {B0_3rd:.2f} ± {perr_3rd[1]:.2f} GPa")
            print(f"B₀' = {B0_prime_3rd:.3f} ± {perr_3rd[2]:.3f}")
            print(f"R² = {r_squared_3rd:.6f}")
            print(f"RMSE = {rmse_3rd:.4f} GPa")

        except Exception as e:
            print(f"⚠️ {phase_name} - 3rd order BM fitting failed: {e}")
            results['3rd_order'] = None

        return results

    def fit_all_phases(self):
        """
        Fit both original and new phases

        Returns:
        --------
        tuple : (results_original, results_new)
        """
        if self.V_original is None or self.P_original is None:
            print("❌ Error: No data loaded. Please load data first.")
            return None, None

        print("\n🔧 Starting Birch-Murnaghan equation fitting...")

        self.results_original = self.fit_single_phase(
            self.V_original, self.P_original, "Original Phase"
        )

        self.results_new = self.fit_single_phase(
            self.V_new, self.P_new, "New Phase"
        )

        return self.results_original, self.results_new

    # ==================== Visualization ====================

    def plot_pv_curves(self, save_path=None):
        """
        Plot P-V curves and fitting results

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        """
        if self.results_original is None or self.results_new is None:
            print("❌ Error: No fitting results available. Please run fit_all_phases() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Birch-Murnaghan Equation Fitting for P-V Curves',
                     fontsize=16, fontweight='bold')

        # Original phase - 2nd order BM
        ax = axes[0, 0]
        ax.scatter(self.V_original, self.P_original, s=80, c='blue', marker='o',
                   label='Experimental Data (Original Phase)', alpha=0.7, edgecolors='black')
        if self.results_original['2nd_order'] is not None:
            V_fit = np.linspace(self.V_original.min()*0.95, self.V_original.max()*1.05, 200)
            P_fit = self.birch_murnaghan_2nd(V_fit,
                                         self.results_original['2nd_order']['V0'],
                                         self.results_original['2nd_order']['B0'])
            ax.plot(V_fit, P_fit, 'r-', linewidth=2.5, label='2nd Order BM Fit', alpha=0.8)

            textstr = f"$V_0$ = {self.results_original['2nd_order']['V0']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {self.results_original['2nd_order']['B0']:.2f} GPa\n"
            textstr += f"$B_0'$ = 4.0 (fixed)\n"
            textstr += f"$R^2$ = {self.results_original['2nd_order']['R_squared']:.6f}"
            ax.text(0.68, 0.65, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax.set_ylabel('Pressure P (GPa)', fontsize=12)
        ax.set_title('Original Phase - 2nd Order BM Equation', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Original phase - 3rd order BM
        ax = axes[0, 1]
        ax.scatter(self.V_original, self.P_original, s=80, c='blue', marker='o',
                   label='Experimental Data (Original Phase)', alpha=0.7, edgecolors='black')
        if self.results_original['3rd_order'] is not None:
            V_fit = np.linspace(self.V_original.min()*0.95, self.V_original.max()*1.05, 200)
            P_fit = self.birch_murnaghan_3rd(V_fit,
                                         self.results_original['3rd_order']['V0'],
                                         self.results_original['3rd_order']['B0'],
                                         self.results_original['3rd_order']['B0_prime'])
            ax.plot(V_fit, P_fit, 'g-', linewidth=2.5, label='3rd Order BM Fit', alpha=0.8)

            textstr = f"$V_0$ = {self.results_original['3rd_order']['V0']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {self.results_original['3rd_order']['B0']:.2f} GPa\n"
            textstr += f"$B_0'$ = {self.results_original['3rd_order']['B0_prime']:.3f}\n"
            textstr += f"$R^2$ = {self.results_original['3rd_order']['R_squared']:.6f}"
            ax.text(0.68, 0.65, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightgreen', alpha=0.5))

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax.set_ylabel('Pressure P (GPa)', fontsize=12)
        ax.set_title('Original Phase - 3rd Order BM Equation', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # New phase - 2nd order BM
        ax = axes[1, 0]
        ax.scatter(self.V_new, self.P_new, s=80, c='red', marker='s',
                   label='Experimental Data (New Phase)', alpha=0.7, edgecolors='black')
        if self.results_new['2nd_order'] is not None:
            V_fit = np.linspace(self.V_new.min()*0.95, self.V_new.max()*1.05, 200)
            P_fit = self.birch_murnaghan_2nd(V_fit,
                                         self.results_new['2nd_order']['V0'],
                                         self.results_new['2nd_order']['B0'])
            ax.plot(V_fit, P_fit, 'r-', linewidth=2.5, label='2nd Order BM Fit', alpha=0.8)

            textstr = f"$V_0$ = {self.results_new['2nd_order']['V0']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {self.results_new['2nd_order']['B0']:.2f} GPa\n"
            textstr += f"$B_0'$ = 4.0 (fixed)\n"
            textstr += f"$R^2$ = {self.results_new['2nd_order']['R_squared']:.6f}"
            ax.text(0.68, 0.65, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax.set_ylabel('Pressure P (GPa)', fontsize=12)
        ax.set_title('New Phase - 2nd Order BM Equation', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # New phase - 3rd order BM
        ax = axes[1, 1]
        ax.scatter(self.V_new, self.P_new, s=80, c='red', marker='s',
                   label='Experimental Data (New Phase)', alpha=0.7, edgecolors='black')
        if self.results_new['3rd_order'] is not None:
            V_fit = np.linspace(self.V_new.min()*0.95, self.V_new.max()*1.05, 200)
            P_fit = self.birch_murnaghan_3rd(V_fit,
                                         self.results_new['3rd_order']['V0'],
                                         self.results_new['3rd_order']['B0'],
                                         self.results_new['3rd_order']['B0_prime'])
            ax.plot(V_fit, P_fit, 'g-', linewidth=2.5, label='3rd Order BM Fit', alpha=0.8)

            textstr = f"$V_0$ = {self.results_new['3rd_order']['V0']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {self.results_new['3rd_order']['B0']:.2f} GPa\n"
            textstr += f"$B_0'$ = {self.results_new['3rd_order']['B0_prime']:.3f}\n"
            textstr += f"$R^2$ = {self.results_new['3rd_order']['R_squared']:.6f}"
            ax.text(0.68, 0.65, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightgreen', alpha=0.5))

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax.set_ylabel('Pressure P (GPa)', fontsize=12)
        ax.set_title('New Phase - 3rd Order BM Equation', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ P-V curve figure saved to: {save_path}")

        plt.show()

    def plot_residuals(self, save_path=None):
        """
        Plot fitting residuals analysis

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        """
        if self.results_original is None or self.results_new is None:
            print("❌ Error: No fitting results available. Please run fit_all_phases() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Fitting Residuals Analysis', fontsize=16, fontweight='bold')

        # Original phase - 2nd order BM residuals
        ax = axes[0, 0]
        if self.results_original['2nd_order'] is not None:
            residuals = self.P_original - self.results_original['2nd_order']['fitted_P']
            ax.scatter(self.V_original, residuals, s=60, c='blue', marker='o', alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
            ax.set_ylabel('Residuals (GPa)', fontsize=11)
            ax.set_title('Original Phase - 2nd Order BM Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            textstr = f"RMSE = {self.results_original['2nd_order']['RMSE']:.4f} GPa"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

        # Original phase - 3rd order BM residuals
        ax = axes[0, 1]
        if self.results_original['3rd_order'] is not None:
            residuals = self.P_original - self.results_original['3rd_order']['fitted_P']
            ax.scatter(self.V_original, residuals, s=60, c='blue', marker='o', alpha=0.7)
            ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
            ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
            ax.set_ylabel('Residuals (GPa)', fontsize=11)
            ax.set_title('Original Phase - 3rd Order BM Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            textstr = f"RMSE = {self.results_original['3rd_order']['RMSE']:.4f} GPa"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightgreen', alpha=0.5))

        # New phase - 2nd order BM residuals
        ax = axes[1, 0]
        if self.results_new['2nd_order'] is not None:
            residuals = self.P_new - self.results_new['2nd_order']['fitted_P']
            ax.scatter(self.V_new, residuals, s=60, c='red', marker='s', alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
            ax.set_ylabel('Residuals (GPa)', fontsize=11)
            ax.set_title('New Phase - 2nd Order BM Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            textstr = f"RMSE = {self.results_new['2nd_order']['RMSE']:.4f} GPa"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

        # New phase - 3rd order BM residuals
        ax = axes[1, 1]
        if self.results_new['3rd_order'] is not None:
            residuals = self.P_new - self.results_new['3rd_order']['fitted_P']
            ax.scatter(self.V_new, residuals, s=60, c='red', marker='s', alpha=0.7)
            ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
            ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
            ax.set_ylabel('Residuals (GPa)', fontsize=11)
            ax.set_title('New Phase - 3rd Order BM Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            textstr = f"RMSE = {self.results_new['3rd_order']['RMSE']:.4f} GPa"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Residuals figure saved to: {save_path}")

        plt.show()

    # ==================== Export Results ====================

    def save_results_to_csv(self, output_path):
        """
        Save fitting results to CSV file

        Parameters:
        -----------
        output_path : str
            Path to save the CSV file

        Returns:
        --------
        df_summary : DataFrame
            Summary dataframe of fitting parameters
        """
        if self.results_original is None or self.results_new is None:
            print("❌ Error: No fitting results available. Please run fit_all_phases() first.")
            return None

        summary_data = []

        for phase_name, results in [('Original Phase', self.results_original),
                                    ('New Phase', self.results_new)]:
            for order in ['2nd_order', '3rd_order']:
                if results[order] is not None:
                    row = {
                        'Phase': phase_name,
                        'Fitting Order': '2nd Order' if order == '2nd_order' else '3rd Order',
                        'V₀ (Å³/atom)': f"{results[order]['V0']:.6f}",
                        'V₀ Error': f"{results[order]['V0_err']:.6f}",
                        'B₀ (GPa)': f"{results[order]['B0']:.4f}",
                        'B₀ Error': f"{results[order]['B0_err']:.4f}",
                        "B₀'": f"{results[order]['B0_prime']:.6f}",
                        "B₀' Error": f"{results[order]['B0_prime_err']:.6f}",
                        'R²': f"{results[order]['R_squared']:.8f}",
                        'RMSE (GPa)': f"{results[order]['RMSE']:.6f}"
                    }
                    summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ Fitting parameters saved to: {output_path}")

        return df_summary

    # ==================== Complete Analysis Workflow ====================

    def analyze(self, original_csv, new_csv, output_dir=None):
        """
        Complete analysis workflow: load data, fit, plot, and save results

        Parameters:
        -----------
        original_csv : str
            Path to CSV file containing original phase data
        new_csv : str
            Path to CSV file containing new phase data
        output_dir : str, optional
            Directory to save output files. If None, files are not saved.

        Returns:
        --------
        results : dict
            Dictionary containing fitting results for both phases
        """
        print("\n" + "="*80)
        print("Birch-Murnaghan Equation Fitting for P-V Curves")
        print("="*80)

        # Load data
        print(f"\n📂 Loading data files...")
        if not self.load_data_from_csv(original_csv, new_csv):
            return None

        # Display data overview
        print(f"\n📊 Data Overview:")
        print(f"   Original phase volume range: {self.V_original.min():.4f} - {self.V_original.max():.4f} Å³/atom")
        print(f"   Original phase pressure range: {self.P_original.min():.2f} - {self.P_original.max():.2f} GPa")
        print(f"   New phase volume range: {self.V_new.min():.4f} - {self.V_new.max():.4f} Å³/atom")
        print(f"   New phase pressure range: {self.P_new.min():.2f} - {self.P_new.max():.2f} GPa")

        # Perform fitting
        results_orig, results_new = self.fit_all_phases()

        if results_orig is None or results_new is None:
            return None

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Plot and save figures
            print(f"\n📈 Plotting P-V curves...")
            pv_curve_path = os.path.join(output_dir, 'BM_fitting_results.png')
            self.plot_pv_curves(save_path=pv_curve_path)

            print(f"\n📉 Plotting residuals...")
            residuals_path = os.path.join(output_dir, 'BM_fitting_residuals.png')
            self.plot_residuals(save_path=residuals_path)

            # Save results to CSV
            print(f"\n💾 Saving fitting parameters...")
            csv_path = os.path.join(output_dir, 'BM_fitting_parameters.csv')
            self.save_results_to_csv(csv_path)

            print(f"\n{'='*80}")
            print("✨ All tasks completed!")
            print(f"{'='*80}")
            print(f"📁 Output directory: {output_dir}")
            print(f"   - BM_fitting_results.png : P-V curve fitting plots")
            print(f"   - BM_fitting_residuals.png : Residual analysis plots")
            print(f"   - BM_fitting_parameters.csv : Fitting parameters summary table")
            print(f"{'='*80}\n")
        else:
            # Just display plots without saving
            print(f"\n📈 Plotting P-V curves...")
            self.plot_pv_curves()

            print(f"\n📉 Plotting residuals...")
            self.plot_residuals()

        return {
            'original_phase': results_orig,
            'new_phase': results_new
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example: Basic usage
    fitter = BirchMurnaghanFitter()

    # Set data directory
    data_dir = r"D:\HEPS\ID31\dioptas_data\Al0\fit_output"
    orig_file = os.path.join(data_dir, "all_results_original_peaks_lattice.csv")
    new_file = os.path.join(data_dir, "all_results_new_peaks_lattice.csv")
    output_dir = os.path.join(data_dir, "BM_fitting_output")

    # Run complete analysis
    results = fitter.analyze(orig_file, new_file, output_dir)