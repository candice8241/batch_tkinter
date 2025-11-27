# -*- coding: utf-8 -*-
"""
X-ray Diffraction Analysis Tool - Class-based Implementation
Created on Thu Nov 13 14:30:34 2025

@author: 16961
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import warnings
import re
warnings.filterwarnings('ignore')


class XRayDiffractionAnalyzer:
    """
    X-ray Diffraction Analysis Tool for phase transition identification and lattice parameter fitting.

    This class provides comprehensive analysis of X-ray diffraction data including:
    - Phase transition detection
    - Peak tracking across pressure points
    - Lattice parameter fitting for various crystal systems
    - Atomic volume calculations
    """

    # Crystal system definitions
    CRYSTAL_SYSTEMS = {
        'cubic_FCC': {
            'name': 'FCC',
            'min_peaks': 1,
            'atoms_per_cell': 4,
            'hkl_list': [
                (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2),
                (4,0,0), (3,3,1), (4,2,0), (4,2,2), (3,3,3),
                (5,1,1), (4,4,0), (5,3,1), (6,0,0), (6,2,0),
                (5,3,3), (6,2,2), (4,4,4), (5,5,1), (6,4,0)
            ]
        },
        'cubic_BCC': {
            'name': 'BCC',
            'min_peaks': 1,
            'atoms_per_cell': 2,
            'hkl_list': [
                (1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0),
                (2,2,2), (3,2,1), (4,0,0), (3,3,0), (4,1,1),
                (3,3,2), (4,2,0), (4,2,2), (3,3,3), (5,1,0),
                (4,3,1), (5,2,1), (4,4,0), (5,3,0), (6,0,0)
            ]
        },
        'cubic_SC': {
            'name': 'SC',
            'min_peaks': 1,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0),
                (2,1,1), (2,2,0), (2,2,1), (3,0,0), (3,1,0),
                (3,1,1), (2,2,2), (3,2,0), (3,2,1), (4,0,0),
                (4,1,0), (3,3,0), (4,1,1), (3,3,1), (4,2,0)
            ]
        },
        'Hexagonal': {
            'name': 'HCP',
            'min_peaks': 2,
            'atoms_per_cell': 2,
            'hkl_list': [
                (1,0,0), (1,0,1), (1,0,2), (1,1,0),
                (1,0,3), (2,0,0), (1,1,2), (2,0,1), (0,0,4),
                (2,0,2), (1,0,4), (2,0,3), (2,1,0), (2,1,1),
                (2,0,4), (2,1,2), (3,0,0), (2,1,3), (2,2,0)
            ]
        },
        'Tetragonal': {
            'name': 'Tetragonal',
            'min_peaks': 2,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,0,1), (1,1,0), (1,0,1), (1,1,1),
                (2,0,0), (2,1,0), (0,0,2), (2,1,1), (2,0,1),
                (2,2,0), (2,1,2), (3,0,0), (2,2,1), (3,1,0),
                (2,0,2), (3,1,1), (2,2,2), (3,2,0), (3,0,1)
            ]
        },
        'Orthorhombic': {
            'name': 'Orthorhombic',
            'min_peaks': 3,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                (0,1,1), (1,1,1), (2,0,0), (2,1,0), (2,0,1),
                (1,2,0), (0,2,0), (1,2,1), (0,2,1), (2,1,1),
                (2,2,0), (2,0,2), (0,0,2), (2,2,1), (3,0,0)
            ]
        },
        'Monoclinic': {
            'name': 'Monoclinic',
            'min_peaks': 4,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                (0,1,1), (1,-1,0), (1,0,-1), (1,1,1), (2,0,0),
                (1,-1,1), (2,1,0), (0,2,0), (2,0,1), (1,2,0),
                (0,0,2), (2,1,1), (1,1,-1), (2,-1,0), (2,0,-1)
            ]
        },
        'Triclinic': {
            'name': 'Triclinic',
            'min_peaks': 6,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                (0,1,1), (1,-1,0), (1,0,-1), (0,1,-1), (1,1,1),
                (1,-1,1), (1,1,-1), (2,0,0), (0,2,0), (0,0,2),
                (2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2)
            ]
        }
    }

    def __init__(self, wavelength=0.4133, n_pressure_points=4):
        """
        Initialize XRayDiffractionAnalyzer

        Parameters:
            wavelength (float): X-ray wavelength in Angstroms (default: 0.4133 Å)
            n_pressure_points (int): Number of pressure points required for stable new peak determination
        """
        self.wavelength = wavelength
        # Fixed tolerances for peak identification and tracking
        self.peak_tolerance_1 = 0.3
        self.peak_tolerance_2 = 0.4
        self.peak_tolerance_3 = 0.01
        self.n_pressure_points = n_pressure_points

        # Data storage
        self.pressure_data = None
        self.transition_pressure = None
        self.before_pressures = []
        self.after_pressures = []
        self.original_peak_dataset = None
        self.tracked_new_peaks = None
        self.original_results = None
        self.new_results = None

    # ==================== Utility Functions ====================

    @staticmethod
    def two_theta_to_d(two_theta, wavelength):
        """Convert 2theta angle to d-spacing"""
        theta_rad = np.deg2rad(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta_rad))

    @staticmethod
    def d_to_two_theta(d, wavelength):
        """Convert d-spacing to 2theta angle"""
        sin_theta = wavelength / (2.0 * d)
        if sin_theta > 1.0 or sin_theta < -1.0:
            return None
        theta_rad = np.arcsin(sin_theta)
        return np.rad2deg(2.0 * theta_rad)

    @staticmethod
    def calculate_d_cubic(hkl, a):
        """Calculate d-spacing for cubic crystal system"""
        h, k, l = hkl
        return a / np.sqrt(h**2 + k**2 + l**2)

    @staticmethod
    def calculate_d_hexagonal(hkl, a, c):
        """Calculate d-spacing for hexagonal crystal system"""
        h, k, l = hkl
        return 1.0 / np.sqrt(4.0/3.0 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)

    @staticmethod
    def calculate_d_tetragonal(hkl, a, c):
        """Calculate d-spacing for tetragonal crystal system"""
        h, k, l = hkl
        return 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)

    @staticmethod
    def calculate_d_orthorhombic(hkl, a, b, c):
        """Calculate d-spacing for orthorhombic crystal system"""
        h, k, l = hkl
        return 1.0 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)

    @staticmethod
    def calculate_d_monoclinic(hkl, a, b, c, beta):
        """Calculate d-spacing for monoclinic crystal system"""
        h, k, l = hkl
        beta_rad = np.deg2rad(beta)
        sin_beta = np.sin(beta_rad)
        cos_beta = np.cos(beta_rad)
        term = (h**2 / a**2 + k**2 * sin_beta**2 / b**2 + l**2 / c**2
                - 2*h*l*cos_beta / (a*c)) / sin_beta**2
        return 1.0 / np.sqrt(term)

    @staticmethod
    def calculate_cell_volume_cubic(a):
        """Calculate unit cell volume for cubic system"""
        return a**3

    @staticmethod
    def calculate_cell_volume_hexagonal(a, c):
        """Calculate unit cell volume for hexagonal system"""
        return np.sqrt(3) / 2 * a**2 * c

    @staticmethod
    def calculate_cell_volume_tetragonal(a, c):
        """Calculate unit cell volume for tetragonal system"""
        return a**2 * c

    @staticmethod
    def calculate_cell_volume_orthorhombic(a, b, c):
        """Calculate unit cell volume for orthorhombic system"""
        return a * b * c

    @staticmethod
    def calculate_cell_volume_monoclinic(a, b, c, beta):
        """Calculate unit cell volume for monoclinic system"""
        beta_rad = np.deg2rad(beta)
        return a * b * c * np.sin(beta_rad)

    # ==================== Data Reading ====================

    def read_pressure_peak_data(self, csv_path):
        """
        Read CSV file and extract pressure points and peak positions

        Parameters:
            csv_path (str): Path to CSV file

        Returns:
            dict: Dictionary with pressure values (GPa) as keys and peak position lists (2theta) as values
        """
        df = pd.read_csv(csv_path)

        if 'File' not in df.columns or 'Center' not in df.columns:
            raise ValueError("CSV file must contain 'File' and 'Center' columns")

        pressure_data = {}

        for idx, row in df.iterrows():
            # Check if this is a blank row
            if pd.isna(row['File']) or row['File'] == '':
                continue

            # Extract pressure value
            try:
                file_str = str(row['File'])
                numbers = re.findall(r'[-+]?\d*\.?\d+', file_str)
                if numbers:
                    pressure = float(numbers[0])
                else:
                    pressure = float(file_str)
            except:
                print(f"Warning: Cannot parse pressure value: {row['File']}")
                continue

            # Extract peak position
            try:
                peak_position = float(row['Center'])
            except:
                print(f"Warning: Cannot parse peak position: {row['Center']}")
                continue

            if pressure not in pressure_data:
                pressure_data[pressure] = []
            pressure_data[pressure].append(peak_position)

        # Sort peak positions for each pressure point
        for pressure in pressure_data:
            pressure_data[pressure] = sorted(pressure_data[pressure])

        self.pressure_data = pressure_data
        return pressure_data

    # ==================== Phase Transition Identification ====================

    def find_phase_transition_point(self, pressure_data=None, tolerance=None):
        """
        Identify phase transition point

        Parameters:
            pressure_data (dict, optional): Pressure-peak data dictionary
            tolerance (float, optional): Peak position tolerance (degrees)

        Returns:
            tuple: (transition_pressure, before_pressures, after_pressures)
        """
        if pressure_data is None:
            pressure_data = self.pressure_data
        if tolerance is None:
            tolerance = self.peak_tolerance_1

        sorted_pressures = sorted(pressure_data.keys())

        if len(sorted_pressures) < 2:
            print("Warning: Less than 2 pressure points, cannot identify phase transition")
            return None, sorted_pressures, []

        for i in range(1, len(sorted_pressures)):
            prev_pressure = sorted_pressures[i - 1]
            curr_pressure = sorted_pressures[i]

            prev_peaks = pressure_data[prev_pressure]
            curr_peaks = pressure_data[curr_pressure]

            tolerance_windows = [(p - tolerance, p + tolerance) for p in prev_peaks]

            has_new_peak = False
            for peak in curr_peaks:
                in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
                if not in_any_window:
                    has_new_peak = True
                    break

            if has_new_peak:
                print(f"\n>>> Phase transition detected at: {curr_pressure:.2f} GPa")
                self.transition_pressure = curr_pressure
                self.before_pressures = sorted_pressures[:i]
                self.after_pressures = sorted_pressures[i:]
                return curr_pressure, self.before_pressures, self.after_pressures

        print("\n>>> No obvious phase transition detected")
        self.transition_pressure = None
        self.before_pressures = sorted_pressures
        self.after_pressures = []
        return None, sorted_pressures, []

    # ==================== Peak Collection ====================

    def collect_tracked_new_peaks(self, pressure_data=None, transition_pressure=None,
                                   after_pressures=None, new_peaks_ref=None, tolerance=None,
                                   output_csv=None):
        """
        Track specified new peaks starting from phase transition pressure and export to CSV.

        Parameters:
            pressure_data (dict, optional): Pressure-peak data dictionary
            transition_pressure (float, optional): Phase transition pressure
            after_pressures (list, optional): List of pressures after transition
            new_peaks_ref (list, optional): Reference list of new peaks to track
            tolerance (float, optional): Peak position tolerance (degrees)
            output_csv (str, optional): Output CSV file path for new peaks dataset

        Returns:
            tuple: (stable_count, tracked_peaks_dict)
        """
        if pressure_data is None:
            pressure_data = self.pressure_data
        if tolerance is None:
            tolerance = self.peak_tolerance_2

        tracked_peaks_dict = {}
        peak_occurrences = {peak: 0 for peak in new_peaks_ref}

        # Track new peaks across all pressures after transition
        for pressure in after_pressures:
            current_peaks = pressure_data[pressure]
            matched_peaks = []

            for new_peak in new_peaks_ref:
                lower = new_peak - tolerance
                upper = new_peak + tolerance

                matches = [p for p in current_peaks if lower <= p <= upper]
                if matches:
                    mean_match = np.mean(matches)
                    matched_peaks.append(mean_match)
                    peak_occurrences[new_peak] += 1

            if matched_peaks:
                tracked_peaks_dict[pressure] = matched_peaks

        # Count stable peaks
        stable_count = sum(1 for count in peak_occurrences.values()
                           if count >= self.n_pressure_points)

        self.tracked_new_peaks = tracked_peaks_dict

        # Export new peaks dataset to CSV
        if output_csv is not None:
            self._export_peaks_to_csv(tracked_peaks_dict, output_csv,
                                     dataset_type="New Peaks")
            print(f"✓ New peaks dataset saved to: {output_csv}")

        return stable_count, tracked_peaks_dict

    def build_original_peak_dataset(self, pressure_data=None, tracked_new_peak_dataset=None,
                                     tolerance=None, output_csv=None):
        """
        Build original peak dataset based on new peak dataset and export to CSV.

        Parameters:
            pressure_data (dict, optional): Pressure-peak data dictionary
            tracked_new_peak_dataset (dict, optional): Tracked new peaks dataset
            tolerance (float, optional): Peak position tolerance (degrees)
            output_csv (str, optional): Output CSV file path for original peaks dataset

        Returns:
            dict: Original peak dataset
        """
        if pressure_data is None:
            pressure_data = self.pressure_data
        if tracked_new_peak_dataset is None:
            tracked_new_peak_dataset = self.tracked_new_peaks
        if tolerance is None:
            tolerance = self.peak_tolerance_3

        original_peak_dataset = {}

        # Identify original peaks by excluding new peaks
        for pressure, all_peaks in pressure_data.items():
            new_peaks = tracked_new_peak_dataset.get(pressure, [])
            original_peaks = []

            for peak in all_peaks:
                is_new = any(abs(peak - new_peak) <= tolerance for new_peak in new_peaks)
                if not is_new:
                    original_peaks.append(peak)

            original_peak_dataset[pressure] = {
                'original_peaks': original_peaks,
                'count': len(original_peaks)
            }

        self.original_peak_dataset = original_peak_dataset

        # Export original peaks dataset to CSV
        if output_csv is not None:
            # Convert format for export
            peaks_dict_for_export = {p: data['original_peaks']
                                    for p, data in original_peak_dataset.items()}
            self._export_peaks_to_csv(peaks_dict_for_export, output_csv,
                                     dataset_type="Original Peaks")
            print(f"✓ Original peaks dataset saved to: {output_csv}")

        return original_peak_dataset

    @staticmethod

    def _export_peaks_to_csv(peaks_dict, output_path, dataset_type="Peaks"):
        """
        Export peaks dataset to CSV file in vertical (long) format with empty rows between pressure points.
    
        Parameters:
            peaks_dict (dict): Dictionary with pressure as keys and peak lists as values
            output_path (str): Output CSV file path
            dataset_type (str): Type of dataset (for header)
        """
        if not peaks_dict:
            print(f"Warning: No {dataset_type} data to export")
            return
    
        data_rows = []
    
        for pressure in sorted(peaks_dict.keys()):
            peaks = peaks_dict[pressure]
            for i, peak in enumerate(peaks, start=1):
                data_rows.append([i, peak, pressure])
            data_rows.append([None, None, None])  # 插入空行
    
        df = pd.DataFrame(data_rows, columns=["Peak #", "Center", "File"])
        df.to_csv(output_path, index=False)


    # ==================== Lattice Parameter Fitting ====================

    def fit_lattice_parameters_cubic(self, peak_dataset, crystal_system_key):
        """Fit lattice parameters for cubic crystal systems"""
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) == 0:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a = params[0]
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_cubic(hkl, a)
                    errors.append(d_obs[i] - d_calc)
                return errors

            a_init = d_obs[0] * np.sqrt(sum(x**2 for x in matched_hkl[0]))
            result = least_squares(residuals, [a_init], bounds=([0], [np.inf]))
            a_fitted = result.x[0]

            V_cell = self.calculate_cell_volume_cubic(a_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

            print(f"Pressure: {pressure:.2f} GPa")
            print(f"  Lattice parameter a = {a_fitted:.6f} Å")
            print(f"  Unit cell volume V = {V_cell:.6f} Å³")
            print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

        return results

    def fit_lattice_parameters_hexagonal(self, peak_dataset, crystal_system_key):
        """Fit lattice parameters for hexagonal crystal systems"""
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) < 2:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a, c = params
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_hexagonal(hkl, a, c)
                    errors.append(d_obs[i] - d_calc)

                # Soft constraint for c/a ratio ~ 1.633
                target_ratio = 1.633
                ratio = c / a
                penalty_weight = 0.1
                penalty = penalty_weight * (ratio - target_ratio)
                errors.append(penalty)

                return errors

            a_init = 3.0
            c_init = 5.0

            result = least_squares(residuals, [a_init, c_init],
                                  bounds=([0, 0], [np.inf, np.inf]))
            a_fitted, c_fitted = result.x

            V_cell = self.calculate_cell_volume_hexagonal(a_fitted, c_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'c': c_fitted,
                'c/a': c_fitted / a_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

            print(f"Pressure: {pressure:.2f} GPa")
            print(f"  Lattice parameter a = {a_fitted:.6f} Å")
            print(f"  Lattice parameter c = {c_fitted:.6f} Å")
            print(f"  c/a ratio = {c_fitted/a_fitted:.6f}")
            print(f"  Unit cell volume V = {V_cell:.6f} Å³")
            print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

        return results

    def fit_lattice_parameters_tetragonal(self, peak_dataset, crystal_system_key):
        """Fit lattice parameters for tetragonal crystal systems"""
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) < 2:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a, c = params
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_tetragonal(hkl, a, c)
                    errors.append(d_obs[i] - d_calc)
                return errors

            a_init = 3.0
            c_init = 4.0

            result = least_squares(residuals, [a_init, c_init],
                                  bounds=([0, 0], [np.inf, np.inf]))
            a_fitted, c_fitted = result.x

            V_cell = self.calculate_cell_volume_tetragonal(a_fitted, c_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'c': c_fitted,
                'c/a': c_fitted / a_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

            print(f"Pressure: {pressure:.2f} GPa")
            print(f"  Lattice parameter a = {a_fitted:.6f} Å")
            print(f"  Lattice parameter c = {c_fitted:.6f} Å")
            print(f"  c/a ratio = {c_fitted/a_fitted:.6f}")
            print(f"  Unit cell volume V = {V_cell:.6f} Å³")
            print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

        return results

    def fit_lattice_parameters_orthorhombic(self, peak_dataset, crystal_system_key):
        """Fit lattice parameters for orthorhombic crystal systems"""
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) < 3:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a, b, c = params
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_orthorhombic(hkl, a, b, c)
                    errors.append(d_obs[i] - d_calc)
                return errors

            a_init = 3.0
            b_init = 4.0
            c_init = 5.0

            result = least_squares(residuals, [a_init, b_init, c_init],
                                  bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            a_fitted, b_fitted, c_fitted = result.x

            V_cell = self.calculate_cell_volume_orthorhombic(a_fitted, b_fitted, c_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'b': b_fitted,
                'c': c_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

            print(f"Pressure: {pressure:.2f} GPa")
            print(f"  Lattice parameter a = {a_fitted:.6f} Å")
            print(f"  Lattice parameter b = {b_fitted:.6f} Å")
            print(f"  Lattice parameter c = {c_fitted:.6f} Å")
            print(f"  Unit cell volume V = {V_cell:.6f} Å³")
            print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

        return results

    def fit_lattice_parameters(self, peak_dataset, crystal_system_key):
        """
        Main function to fit lattice parameters based on crystal system

        Parameters:
            peak_dataset (dict): Peak dataset dictionary
            crystal_system_key (str): Crystal system key

        Returns:
            dict: Fitting results including lattice parameters and volumes
        """
        system_type = crystal_system_key.split('_')[0]

        print(f"\n{'='*60}")
        print(f"Fitting Lattice Parameters for {self.CRYSTAL_SYSTEMS[crystal_system_key]['name']}")
        print(f"{'='*60}\n")

        if system_type == 'cubic':
            return self.fit_lattice_parameters_cubic(peak_dataset, crystal_system_key)
        elif crystal_system_key == 'Hexagonal':
            return self.fit_lattice_parameters_hexagonal(peak_dataset, crystal_system_key)
        elif crystal_system_key == 'Tetragonal':
            return self.fit_lattice_parameters_tetragonal(peak_dataset, crystal_system_key)
        elif crystal_system_key == 'Orthorhombic':
            return self.fit_lattice_parameters_orthorhombic(peak_dataset, crystal_system_key)
        else:
            print(f"Warning: Fitting for {crystal_system_key} not yet implemented")
            return {}

    # ==================== User Interaction ====================

    @staticmethod
    def select_crystal_system(label=""):
        """Interactive crystal system selection"""
        print(f"\nSelect crystal system for {label}:")
        print("[1] Face-Centered Cubic (FCC)")
        print("[2] Body-Centered Cubic (BCC)")
        print("[3] Simple Cubic (SC)")
        print("[4] Hexagonal Close-Packed (HCP)")
        print("[5] Tetragonal")
        print("[6] Orthorhombic")
        print("[7] Monoclinic")
        print("[8] Triclinic")

        choice = input("Enter your choice (1-8): ").strip()

        mapping = {
            "1": "cubic_FCC",
            "2": "cubic_BCC",
            "3": "cubic_SC",
            "4": "Hexagonal",
            "5": "Tetragonal",
            "6": "Orthorhombic",
            "7": "Monoclinic",
            "8": "Triclinic"
        }

        selected = mapping.get(choice)
        if selected:
            print(f"✓ Selected crystal system: {XRayDiffractionAnalyzer.CRYSTAL_SYSTEMS[selected]['name']}")
            return selected
        else:
            print("⚠️ Invalid choice, defaulting to 'cubic_FCC'")
            return "cubic_FCC"

    # ==================== Results Export ====================

    @staticmethod
    def save_lattice_results_to_csv(results, filename, crystal_system_key):
        """Save lattice parameter fitting results to CSV file"""
        if not results:
            print("No results to save.")
            return

        data_rows = []
        for pressure, params in sorted(results.items()):
            row = {'Pressure (GPa)': pressure}
            row.update(params)
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to: {filename}")

    # ==================== Main Analysis Workflow ====================

    def analyze(self, csv_path, original_system=None, new_system=None,
                auto_mode=False):
        """
        Complete analysis workflow

        Parameters:
            csv_path (str): Path to input CSV file
            original_system (str, optional): Crystal system for original peaks
            new_system (str, optional): Crystal system for new peaks
            auto_mode (bool): If True, use provided systems without user interaction

        Returns:
            dict: Analysis results containing original_results and new_results
        """
        print("\n" + "="*60)
        print("X-RAY DIFFRACTION ANALYSIS - PHASE TRANSITION & LATTICE FITTING")
        print("="*60)

        # Step 1: Read data
        try:
            self.read_pressure_peak_data(csv_path)
            print(f"\n✓ Successfully read data from {csv_path}")
            print(f"  Total pressure points: {len(self.pressure_data)}")
        except Exception as e:
            print(f"❌ Error reading CSV file: {e}")
            return None

        # Step 2: Identify phase transition
        print("\n" + "="*60)
        print("PHASE TRANSITION IDENTIFICATION")
        print("="*60)

        transition_pressure, before_pressures, after_pressures = self.find_phase_transition_point()

        if transition_pressure is None:
            print("\nNo phase transition detected. Analyzing as single phase...")

            if not auto_mode:
                system_key = self.select_crystal_system()
            else:
                system_key = original_system or 'cubic_FCC'

            all_data_dict = {p: peaks for p, peaks in self.pressure_data.items()}
            results = self.fit_lattice_parameters(all_data_dict, system_key)

            output_filename = csv_path.replace('.csv', '_lattice_results.csv')
            self.save_lattice_results_to_csv(results, output_filename, system_key)

            return {'single_phase_results': results}

        # Step 3: Collect new peaks and original peaks
        print("\n" + "="*60)
        print("COLLECTING NEW PEAKS AND ORIGINAL PEAKS")
        print("="*60)

        transition_peaks = self.pressure_data[transition_pressure]
        prev_pressure = before_pressures[-1]
        prev_peaks = self.pressure_data[prev_pressure]

        tolerance_windows = [(p - self.peak_tolerance_1, p + self.peak_tolerance_1)
                            for p in prev_peaks]
        new_peaks_at_transition = []

        for peak in transition_peaks:
            in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
            if not in_any_window:
                new_peaks_at_transition.append(peak)

        print(f"\nNew peaks detected at transition: {len(new_peaks_at_transition)}")
        print(f"Positions: {[f'{p:.3f}' for p in new_peaks_at_transition]}")

        # Prepare output file paths for peak datasets
        base_filename = csv_path.replace('.csv', '')
        new_peaks_dataset_csv = f"{base_filename}_new_peaks_dataset.csv"
        original_peaks_dataset_csv = f"{base_filename}_original_peaks_dataset.csv"

        # Track new peaks and export to CSV
        stable_count, tracked_new_peaks = self.collect_tracked_new_peaks(
            self.pressure_data, transition_pressure, after_pressures,
            new_peaks_at_transition, self.peak_tolerance_2,
            output_csv=new_peaks_dataset_csv
        )

        print(f"\nStable new peaks (appearing in ≥{self.n_pressure_points} pressure points): {stable_count}")

        # Build original peaks dataset and export to CSV
        original_peak_dataset = self.build_original_peak_dataset(
            self.pressure_data, tracked_new_peaks, self.peak_tolerance_3,
            output_csv=original_peaks_dataset_csv
        )

        print(f"\nOriginal peaks dataset constructed for {len(original_peak_dataset)} pressure points")

        # Step 4: Select crystal systems
        print("\n" + "="*60)
        print("CRYSTAL SYSTEM SELECTION")
        print("="*60)

        if not auto_mode:
            original_system = self.select_crystal_system("ORIGINAL PEAKS (before transition)")
            new_system = self.select_crystal_system("NEW PEAKS (after transition)")
        else:
            original_system = original_system or 'cubic_FCC'
            new_system = new_system or 'cubic_FCC'
            print(f"✓ Using crystal system for original peaks: {self.CRYSTAL_SYSTEMS[original_system]['name']}")
            print(f"✓ Using crystal system for new peaks: {self.CRYSTAL_SYSTEMS[new_system]['name']}")

        # Step 5: Fit lattice parameters
        print("\n" + "="*60)
        print("FITTING LATTICE PARAMETERS")
        print("="*60)

        print("\n>>> FITTING ORIGINAL PEAKS <<<")
        self.original_results = self.fit_lattice_parameters(original_peak_dataset, original_system)

        print("\n>>> FITTING NEW PEAKS <<<")
        self.new_results = self.fit_lattice_parameters(tracked_new_peaks, new_system)

        # Step 6: Save results
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        original_output = f"{base_filename}_original_peaks_lattice.csv"
        self.save_lattice_results_to_csv(self.original_results, original_output, original_system)

        new_output = f"{base_filename}_new_peaks_lattice.csv"
        self.save_lattice_results_to_csv(self.new_results, new_output, new_system)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nSummary:")
        print(f"  - Phase transition pressure: {transition_pressure:.2f} GPa")
        print(f"  - Original peaks crystal system: {self.CRYSTAL_SYSTEMS[original_system]['name']}")
        print(f"  - New peaks crystal system: {self.CRYSTAL_SYSTEMS[new_system]['name']}")
        print(f"  - New peaks dataset saved to: {new_peaks_dataset_csv}")
        print(f"  - Original peaks dataset saved to: {original_peaks_dataset_csv}")
        print(f"  - Original peaks results saved to: {original_output}")
        print(f"  - New peaks results saved to: {new_output}")
        print("\n" + "="*60 + "\n")

        return {
            'original_results': self.original_results,
            'new_results': self.new_results,
            'transition_pressure': transition_pressure
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Interactive mode example
    analyzer = XRayDiffractionAnalyzer(
        wavelength=0.4133,
        n_pressure_points=4
    )

    csv_path = r'D:\HEPS\ID31\dioptas_data\Al0\fit_output\all_results.csv'
    results = analyzer.analyze(csv_path)