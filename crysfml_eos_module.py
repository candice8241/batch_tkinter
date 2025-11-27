# -*- coding: utf-8 -*-
"""
CrysFML EoS Module - Python Implementation
Comprehensive Equation of State (EoS) Module inspired by CrysFML cfml_eos
Includes multiple EoS models: Murnaghan, Vinet, Tait, Birch-Murnaghan, Natural Strain

Based on:
- Angel, R.J., Alvaro, M., and Gonzalez-Platas, J. (2014)
  "EosFit7c and a Fortran module (library) for equation of state calculations"
  Zeitschrift für Kristallographie - Crystalline Materials, 229(5), 405-419
- ASE (Atomic Simulation Environment) EoS implementation
- pwtools EoS implementation

@author: Integration by candicewang928@gmail.com
Created: 2025-11-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, differential_evolution, least_squares
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

warnings.filterwarnings('ignore', category=RuntimeWarning)


class EoSType(Enum):
    """Enumeration of supported Equation of State types"""
    MURNAGHAN = "murnaghan"
    BIRCH_MURNAGHAN_2ND = "birch_murnaghan_2nd"
    BIRCH_MURNAGHAN_3RD = "birch_murnaghan_3rd"
    BIRCH_MURNAGHAN_4TH = "birch_murnaghan_4th"
    VINET = "vinet"
    TAIT = "tait"
    NATURAL_STRAIN = "natural_strain"


@dataclass
class EoSParameters:
    """
    Data structure to hold EoS parameters (similar to CrysFML's eos_type)

    Attributes:
    -----------
    eos_type : EoSType
        Type of equation of state
    V0 : float
        Zero-pressure volume (Å³/atom)
    V0_err : float
        Error in V0
    B0 : float
        Zero-pressure bulk modulus (GPa)
    B0_err : float
        Error in B0
    B0_prime : float
        First pressure derivative of bulk modulus (dimensionless)
    B0_prime_err : float
        Error in B0_prime
    B0_prime2 : float
        Second pressure derivative of bulk modulus (for 4th order)
    B0_prime2_err : float
        Error in B0_prime2
    E0 : float
        Reference energy (optional, for E-V curves)
    R_squared : float
        Coefficient of determination
    RMSE : float
        Root mean square error
    chi2 : float
        Reduced chi-square statistic (weighted)
    n_data : int
        Number of data points
    """
    eos_type: EoSType
    V0: float = 0.0
    V0_err: float = 0.0
    B0: float = 0.0
    B0_err: float = 0.0
    B0_prime: float = 4.0
    B0_prime_err: float = 0.0
    B0_prime2: float = 0.0
    B0_prime2_err: float = 0.0
    E0: float = 0.0
    R_squared: float = 0.0
    RMSE: float = 0.0
    chi2: float = 0.0
    n_data: int = 0


class CrysFMLEoS:
    """
    Comprehensive Equation of State Calculator

    Provides multiple EoS models and fitting capabilities similar to CrysFML's cfml_eos module
    """

    # ==================== Static Methods: EoS Equations ====================

    @staticmethod
    def murnaghan_pv(V: np.ndarray, V0: float, B0: float, B0_prime: float) -> np.ndarray:
        """
        Murnaghan Equation of State (P-V form)

        P = (B0/B0') * [(V0/V)^B0' - 1]

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            Pressure derivative of bulk modulus

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        P = (B0 / B0_prime) * ((V0 / V)**B0_prime - 1)
        return P

    @staticmethod
    def birch_murnaghan_2nd_pv(V: np.ndarray, V0: float, B0: float) -> np.ndarray:
        """
        2nd order Birch-Murnaghan Equation of State (P-V form)

        This expression mirrors the formulation used inside the CrysFML
        ``eosfit`` module: it is written in terms of the Eulerian strain
        ``f`` instead of the compact ``η`` powers to avoid numerical drift
        when ``V`` is very close to ``V0``.

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        f = 0.5 * ((V0 / V) ** (2 / 3) - 1.0)
        P = 3.0 * B0 * f * (1.0 + 2.0 * f) ** 2.5
        return P

    @staticmethod
    def birch_murnaghan_3rd_pv(V: np.ndarray, V0: float, B0: float, B0_prime: float) -> np.ndarray:
        """
        3rd order Birch-Murnaghan Equation of State (P-V form)

        Rewritten to follow the same Eulerian strain formulation as the
        Fortran ``eosfit`` routines from CrysFML.  Expressing the pressure
        in terms of ``f`` and ``(1+2f)`` matches the linearised ``F-f``
        approach used elsewhere in this module and prevents rounding
        differences between GUI updates and the dedicated fitting path.

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            Pressure derivative of bulk modulus

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        f = 0.5 * ((V0 / V) ** (2 / 3) - 1.0)
        prefactor = 3.0 * B0 * f * (1.0 + 2.0 * f) ** 2.5
        correction = 1.0 + 1.5 * (B0_prime - 4.0) * f
        return prefactor * correction

    @staticmethod
    def birch_murnaghan_4th_pv(V: np.ndarray, V0: float, B0: float,
                                B0_prime: float, B0_prime2: float) -> np.ndarray:
        """
        4th order Birch-Murnaghan Equation of State (P-V form)

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            First pressure derivative of bulk modulus
        B0_prime2 : float
            Second pressure derivative of bulk modulus

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        f = 0.5 * ((V0 / V)**(2/3) - 1)
        P = 3 * B0 * f * (1 + 2*f)**(5/2) * (1 + 3*f*(B0_prime - 4) +
            (3/2)*f**2 * (B0*B0_prime2 + (B0_prime - 4)*(B0_prime - 3) + 35/9))
        return P

    @staticmethod
    def vinet_pv(V: np.ndarray, V0: float, B0: float, B0_prime: float) -> np.ndarray:
        """
        Vinet Equation of State (P-V form)

        From PRB 70, 224107
        Good for very high compression regimes

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            Pressure derivative of bulk modulus

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        eta = (V / V0)**(1/3)
        P = 3 * B0 * (1 - eta) / eta**2 * np.exp(1.5 * (B0_prime - 1) * (1 - eta))
        return P

    @staticmethod
    def tait_pv(V: np.ndarray, V0: float, B0: float, B0_prime: float,
                B0_prime2: float = 0.0) -> np.ndarray:
        """
        Tait Equation of State (P-V form)

        An invertible formulation. With B0_prime2 = 0, reduces to Murnaghan.

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            First pressure derivative of bulk modulus
        B0_prime2 : float
            Second pressure derivative of bulk modulus (default: 0.0)

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        if abs(B0_prime2) < 1e-10:  # Reduce to Murnaghan
            return CrysFMLEoS.murnaghan_pv(V, V0, B0, B0_prime)

        c = B0_prime + B0 * B0_prime2 / B0_prime
        P = (B0 / c) * ((V0 / V)**c - 1)
        return P

    @staticmethod
    def natural_strain_pv(V: np.ndarray, V0: float, B0: float, B0_prime: float) -> np.ndarray:
        """
        Natural Strain Equation of State (P-V form)

        3rd order natural strain EoS

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            Pressure derivative of bulk modulus

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        f_n = np.log(V / V0)
        P = -B0 * f_n * (1 - 0.5 * (B0_prime - 2) * f_n)
        return P

    # ==================== Fitting Methods ====================

    def __init__(self, eos_type: EoSType = EoSType.BIRCH_MURNAGHAN_3RD,
                 V0_bounds: Tuple[float, float] = (0.8, 1.25),
                 B0_bounds: Tuple[float, float] = (50, 500),
                 B0_prime_bounds: Tuple[float, float] = (2.0, 7.0),
                 max_iterations: int = 10000,
                 regularization_strength: float = 1.0):
        """
        Initialize CrysFML EoS Calculator

        Following CrysFML methodology: flexible bounds, quality-first approach

        Parameters:
        -----------
        eos_type : EoSType
            Type of equation of state to use
        V0_bounds : tuple
            Bounds for V0 as multipliers of max experimental volume
        B0_bounds : tuple
            Bounds for bulk modulus B0 in GPa
        B0_prime_bounds : tuple
            Bounds for B0' (2.0-7.0, wide range for flexibility)
        max_iterations : int
            Maximum iterations for curve fitting
        regularization_strength : float
            Regularization strength for B0' constraint (default: 1.0)
            Higher values = stronger constraint toward B0' = 4.0
            Range: 0.1 (weak) to 10.0 (very strong)
        """
        self.eos_type = eos_type
        self.V0_bounds = V0_bounds
        self.B0_bounds = B0_bounds
        self.B0_prime_bounds = B0_prime_bounds
        self.max_iterations = max_iterations
        self.regularization_strength = regularization_strength

        # Select the appropriate EoS function
        self.eos_function = self._get_eos_function()

    def _get_eos_function(self):
        """Get the appropriate EoS function based on type"""
        eos_map = {
            EoSType.MURNAGHAN: self.murnaghan_pv,
            EoSType.BIRCH_MURNAGHAN_2ND: self.birch_murnaghan_2nd_pv,
            EoSType.BIRCH_MURNAGHAN_3RD: self.birch_murnaghan_3rd_pv,
            EoSType.BIRCH_MURNAGHAN_4TH: self.birch_murnaghan_4th_pv,
            EoSType.VINET: self.vinet_pv,
            EoSType.TAIT: self.tait_pv,
            EoSType.NATURAL_STRAIN: self.natural_strain_pv,
        }
        return eos_map.get(self.eos_type)

    def _smart_initial_guess(self, V_data: np.ndarray, P_data: np.ndarray) -> Tuple[float, float, float]:
        """
        Smart initial guess estimation based on data characteristics
        
        Uses physical constraints and data-driven estimation similar to CrysFML approach

        Parameters:
        -----------
        V_data : array
            Volume data
        P_data : array
            Pressure data

        Returns:
        --------
        V0_guess, B0_guess, B0_prime_guess : tuple
            Initial parameter guesses
        """
        # Sort data by pressure for better estimation
        sort_idx = np.argsort(P_data)
        P_sorted = P_data[sort_idx]
        V_sorted = V_data[sort_idx]
        
        # V0: Estimate zero-pressure volume
        # Use the volume at lowest pressure as starting point
        min_P_idx = 0  # After sorting, lowest pressure is at index 0
        V0_guess = V_sorted[min_P_idx]
        
        # If minimum pressure is not near zero, extrapolate using first few points
        if P_sorted[min_P_idx] > 0.5:
            # Linear extrapolation using lowest 2-3 pressure points
            n_points = min(3, len(P_sorted))
            if n_points >= 2:
                # Fit linear P-V relationship for low pressure region
                coeffs = np.polyfit(P_sorted[:n_points], V_sorted[:n_points], 1)
                V0_guess = coeffs[1]  # Intercept at P=0
        
        # Ensure V0 is slightly larger than maximum observed volume
        V0_guess = max(V0_guess, np.max(V_sorted) * 1.005)
        
        # B0: Estimate bulk modulus from initial compressibility
        # K = -V * (dP/dV), use low pressure region
        if len(V_sorted) >= 4:
            # Use first 3-5 points to estimate initial slope
            n_points = min(5, len(V_sorted))
            # Calculate dP/dV using finite differences
            dP = P_sorted[1:n_points] - P_sorted[0:n_points-1]
            dV = V_sorted[1:n_points] - V_sorted[0:n_points-1]
            dP_dV = np.mean(dP / dV)
            
            # B0 = -V * dP/dV at zero pressure
            B0_guess = -V0_guess * dP_dV
            
            # Constrain to reasonable range based on material properties
            B0_guess = np.clip(B0_guess, 80, 300)
        else:
            B0_guess = 150.0
        
        # B0_prime: Use typical value of 4.0
        # Most materials have B0_prime between 3.5 and 5.0
        B0_prime_guess = 4.0

        return V0_guess, B0_guess, B0_prime_guess

    def _birch_murnaghan_f_F_transform(self, V_data: np.ndarray, P_data: np.ndarray, 
                                       V0_estimate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform P-V data to normalized strain-stress (f-F) form
        
        CrysFML KEY METHOD: Linearizes Birch-Murnaghan equation
        
        f = normalized strain = [(V0/V)^(2/3) - 1] / 2
        F = normalized stress = P / [3f(1+2f)^(5/2)]
        
        For BM3: F = B0 [1 + (B0'-4)f]  --> LINEAR in f!
        
        This allows weighted linear regression to get stable B0_prime
        """
        # Calculate Eulerian strain
        x = (V0_estimate / V_data) ** (1.0/3.0)
        f = 0.5 * (x**2 - 1.0)
        
        # Calculate normalized stress (avoid division by zero)
        denominator = 3.0 * f * (1.0 + 2.0*f)**(5.0/2.0)
        
        # Small strain handling
        mask = np.abs(f) > 1e-10
        F = np.zeros_like(P_data)
        F[mask] = P_data[mask] / denominator[mask]
        
        # For very small strains, use limit
        F[~mask] = P_data[~mask] / (3.0 * V0_estimate)
        
        return f, F
    
    def _fit_birch_murnaghan_linear(self, V_data: np.ndarray, P_data: np.ndarray) -> EoSParameters:
        """
        CrysFML method: Two-stage fitting using F-f linearization

        Stage 1: Determine V0 and B0 (2nd order fit or iteration)
        Stage 2: Linear regression F vs f to get B0_prime (stable!)

        This prevents B0_prime divergence because it's determined from a linear fit

        Following CrysFML methodology:
        - Iterative refinement of V0
        - Weighted linear regression in F-f space
        - Full error propagation using variance-covariance matrix
        - Physical constraints on B0' (typically 2-8, centered around 4)
        """
        # Physical bounds on B0' based on literature (Angel et al., Gonzalez-Platas)
        # Most crystalline materials have B0' between 3 and 6
        # We allow slightly wider range (2-8) for flexibility
        B0_PRIME_MIN = 2.0
        B0_PRIME_MAX = 8.0

        # Stage 1: Get initial V0 estimate
        V0_guess, B0_guess, B0_prime_guess = self._smart_initial_guess(V_data, P_data)

        # Iteratively refine V0 using 2nd order BM
        # Start with initial guess (slightly above maximum volume for safety)
        V0_current = max(V0_guess, np.max(V_data) * 1.01)
        V0_history = [V0_current]

        # Sanity check on initial B0 estimate
        if B0_guess < 50 or B0_guess > 500:
            B0_guess = 150.0

        for iteration in range(10):  # Increased iterations for better convergence
            # Transform to f-F space
            f, F = self._birch_murnaghan_f_F_transform(V_data, P_data, V0_current)

            # Check for valid strain range
            if np.max(np.abs(f)) > 0.5:  # Strain too large, adjust V0
                V0_current = np.max(V_data) * 1.05
                V0_history.append(V0_current)
                continue

            # CrysFML weighting scheme: emphasize low-strain data even more
            # Weight = 1 / (strain^2 + epsilon) to avoid singularity
            # Use smaller epsilon for stronger emphasis on low-strain points
            weights = 1.0 / (f**2 + 0.001)
            weights = weights / np.sum(weights) * len(weights)

            # Linear fit: F = a + b*f where b = B0' - 4
            # Use regularized weighted least squares to prevent B0' divergence
            W = np.diag(weights)
            A = np.column_stack([np.ones_like(f), f])

            # Apply Tikhonov regularization to constrain B0' near 4.0
            try:
                # Regularization based on CrysFML methodology: penalize deviations from B0' = 4.0
                # User can adjust regularization_strength parameter
                lambda_reg = self.regularization_strength * np.mean(weights)
                R = np.array([[0.0, 0.0], [0.0, 1.0]])  # Only regularize B0' term

                ATA = A.T @ W @ A
                ATF = A.T @ W @ F
                ATA_reg = ATA + lambda_reg * R  # Add soft constraint

                beta = np.linalg.solve(ATA_reg, ATF)

                B0_fit = beta[0]
                b_fit = beta[1]
                B0_prime_fit = 4.0 + b_fit

                # Physical constraints based on literature
                # B0: 20-800 GPa (very wide, most materials 50-400 GPa)
                if B0_fit < 20 or B0_fit > 800:
                    break  # Stop iteration, use current V0

                # B0': Must be within physical bounds (2-8)
                # This is the KEY improvement following CrysFML/EosFit methodology
                if B0_prime_fit < B0_PRIME_MIN or B0_prime_fit > B0_PRIME_MAX:
                    break  # Stop iteration, use current V0

                # Simple V0 update: minimize RMSE in P-V space
                # Try small adjustments around current V0
                best_V0 = V0_current
                best_rmse = float('inf')

                for delta in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                    V0_test = V0_current * (1.0 + delta)
                    if V0_test > np.max(V_data):  # V0 must be larger than all measured volumes
                        P_test = self.birch_murnaghan_3rd_pv(V_data, V0_test, B0_fit, B0_prime_fit)
                        rmse_test = np.sqrt(np.mean((P_data - P_test)**2))
                        if rmse_test < best_rmse:
                            best_rmse = rmse_test
                            best_V0 = V0_test

                # Damped update
                V0_current = 0.8 * V0_current + 0.2 * best_V0
                V0_history.append(V0_current)

                # Check convergence
                if len(V0_history) > 1:
                    V0_change = abs(V0_history[-1] - V0_history[-2]) / V0_history[-2]
                    if V0_change < 1e-6:  # Convergence threshold
                        break

            except (np.linalg.LinAlgError, RuntimeWarning):
                # If linear solve fails, use initial guess
                break

        # Final fit with refined V0
        f, F = self._birch_murnaghan_f_F_transform(V_data, P_data, V0_current)

        # Final weighted linear regression with improved CrysFML weighting
        # Use smaller epsilon to strongly emphasize low-strain data
        weights = 1.0 / (f**2 + 0.001)
        weights = weights / np.sum(weights) * len(weights)

        W = np.diag(weights)
        A = np.column_stack([np.ones_like(f), f])

        try:
            # CrysFML Method: Regularized least squares with penalty on B0' deviation
            # This prevents B0' from diverging while still allowing it to vary
            #
            # Minimize: ||W(F - A*beta)||² + λ * (b - 0)²
            # where b = beta[1] = B0' - 4, and λ is regularization parameter
            #
            # Physical reasoning: B0' should be close to 4.0 for most materials
            # Regularization keeps it near 4.0 unless data strongly suggests otherwise

            # Regularization parameter: controls how strongly we constrain B0' to ~4.0
            # Larger λ = stronger constraint toward B0' = 4.0
            # User-adjustable via regularization_strength parameter
            lambda_reg = self.regularization_strength * np.mean(weights)

            # Modified normal equations with Tikhonov regularization:
            # (A^T W A + λR) beta = A^T W F
            # where R = [[0, 0], [0, 1]] penalizes deviation in beta[1] only
            R = np.array([[0.0, 0.0], [0.0, 1.0]])  # Only regularize B0' term

            ATA = A.T @ W @ A
            ATF = A.T @ W @ F
            ATA_reg = ATA + lambda_reg * R  # Add regularization

            beta = np.linalg.solve(ATA_reg, ATF)

            # Calculate parameter errors from covariance matrix
            # Following CrysFML error propagation methodology
            F_fit = A @ beta
            residuals_F = F - F_fit

            # Weighted chi-square
            chi2 = np.sum(weights * residuals_F**2)
            dof = len(f) - 2  # degrees of freedom

            # Variance-covariance matrix for B0 and B0_prime
            # Account for regularization in error estimation
            if dof > 0:
                s2 = chi2 / dof
                # Use regularized matrix for proper error propagation
                cov_B0_B0p = s2 * np.linalg.inv(ATA_reg)
                errors_linear = np.sqrt(np.diag(cov_B0_B0p))
            else:
                errors_linear = np.array([0.0, 0.0])

            # Extract parameters
            B0_final = beta[0]
            B0_prime_final = 4.0 + beta[1]

            # Apply physical constraints if B0' is out of bounds
            # If out of bounds, try constrained fit with B0' fixed at nearest bound
            if B0_prime_final < B0_PRIME_MIN:
                # B0' too small, fix at minimum and refit for B0 only
                B0_prime_final = B0_PRIME_MIN
                # Refit with constrained B0': F = B0 [1 + (B0'-4)f]
                # F / [1 + (B0'-4)f] = B0
                constraint_factor = 1.0 + (B0_prime_final - 4.0) * f
                F_constrained = F / constraint_factor
                B0_final = np.sum(weights * F_constrained) / np.sum(weights)
                B0_err = np.sqrt(s2 / np.sum(weights)) if dof > 0 else 0.0
                B0_prime_err = 0.0  # Fixed parameter
            elif B0_prime_final > B0_PRIME_MAX:
                # B0' too large, fix at maximum and refit for B0 only
                B0_prime_final = B0_PRIME_MAX
                constraint_factor = 1.0 + (B0_prime_final - 4.0) * f
                F_constrained = F / constraint_factor
                B0_final = np.sum(weights * F_constrained) / np.sum(weights)
                B0_err = np.sqrt(s2 / np.sum(weights)) if dof > 0 else 0.0
                B0_prime_err = 0.0  # Fixed parameter
            else:
                # Parameters within bounds, use normal errors
                B0_err = errors_linear[0]
                B0_prime_err = errors_linear[1]

            # Estimate V0 error using error propagation
            # The uncertainty in V0 comes from the scatter in the V0 iteration
            if len(V0_history) > 3:
                # Use standard deviation of last few iterations as V0 error estimate
                V0_err = np.std(V0_history[-3:])
            else:
                # Alternative: estimate from residuals
                # σ(V0) ≈ σ(V) * (∂V0/∂V) where σ(V) from residuals
                P_fit = self.birch_murnaghan_3rd_pv(V_data, V0_current, B0_final, B0_prime_final)
                residuals_P = P_data - P_fit
                # Estimate V uncertainty from P residuals: dV/dP ≈ V/B
                V_uncertainty = np.std(residuals_P) * V0_current / B0_final
                V0_err = V_uncertainty * 0.5  # Conservative estimate

            # Calculate R² and RMSE in P-V space (for comparison)
            P_fit = self.birch_murnaghan_3rd_pv(V_data, V0_current, B0_final, B0_prime_final)
            residuals_P = P_data - P_fit
            ss_res = np.sum(residuals_P**2)
            ss_tot = np.sum((P_data - np.mean(P_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals_P**2))

            # Calculate reduced chi-square (CrysFML method)
            # chi2_reduced = (weighted sum of squared residuals) / degrees of freedom
            chi2_reduced = chi2 / dof if dof > 0 else 0.0

            # Create parameter object with full error estimates
            params = EoSParameters(eos_type=self.eos_type)
            params.V0 = V0_current
            params.V0_err = V0_err  # Now properly estimated!
            params.B0 = B0_final
            params.B0_err = B0_err
            params.B0_prime = B0_prime_final
            params.B0_prime_err = B0_prime_err
            params.R_squared = r_squared
            params.RMSE = rmse
            params.chi2 = chi2_reduced
            params.n_data = len(V_data)

            return params

        except np.linalg.LinAlgError:
            return None

    def fit(self, V_data: np.ndarray, P_data: np.ndarray,
            use_smart_guess: bool = True,
            V0_init: Optional[float] = None,
            B0_init: Optional[float] = None,
            B0_prime_init: Optional[float] = None,
            use_weights: bool = True,
            lock_flags: Optional[Dict[str, bool]] = None,
            initial_params: Optional[EoSParameters] = None) -> EoSParameters:
        """
        Fit equation of state to P-V data
        
        Uses CrysFML method for Birch-Murnaghan 3rd order:
        - F-f linearization for stable B0_prime determination
        - Prevents B0_prime divergence through linear regression
        
        Falls back to nonlinear fitting for other EoS types

        Parameters:
        -----------
        V_data : array
            Volume data (Å³/atom)
        P_data : array
            Pressure data (GPa)
        use_smart_guess : bool
            Use smart initial guess based on data (default: True)
        V0_init : float, optional
            Manual initial guess for V0 (overrides smart guess)
        B0_init : float, optional
            Manual initial guess for B0 (overrides smart guess)
        B0_prime_init : float, optional
            Manual initial guess for B0' (overrides smart guess)
        use_weights : bool
            Use weighted fitting (emphasize low pressure data)

        Returns:
        --------
        params : EoSParameters
            Fitted parameters with statistics
        """
        V_data = np.array(V_data)
        P_data = np.array(P_data)
        lock_flags = lock_flags or {}
        any_locked = any(lock_flags.values())

        # CrysFML method: Use F-f linearization for Birch-Murnaghan 3rd order
        # This is THE KEY to preventing B0_prime divergence
        if self.eos_type == EoSType.BIRCH_MURNAGHAN_3RD and not any_locked:
            result = self._fit_birch_murnaghan_linear(V_data, P_data)
            # Check if result is reasonable before returning
            if result is not None:
                # Validate the fit quality with physical constraints
                # B0': Must be within physical range (2-8) based on literature
                if (result.R_squared > 0.5 and  # Reasonable R²
                    result.B0 > 20 and result.B0 < 800 and  # Reasonable B0
                    result.B0_prime >= 2.0 and result.B0_prime <= 8.0):  # Physical B0' bounds
                    return result
                else:
                    # F-f method gave unreasonable results, fall back to nonlinear
                    pass
            # If linear method fails or gives bad results, fall through to nonlinear

        # Nonlinear fitting (for non-BM3 or if linear method failed)
        # Initial guesses
        if use_smart_guess:
            V0_guess, B0_guess, B0_prime_guess = self._smart_initial_guess(V_data, P_data)
        else:
            V0_guess = np.max(V_data) * 1.02
            B0_guess = 150.0
            B0_prime_guess = 4.0

        if initial_params is not None:
            V0_guess = initial_params.V0 or V0_guess
            B0_guess = initial_params.B0 or B0_guess
            B0_prime_guess = initial_params.B0_prime or B0_prime_guess

        # Override with manual values if provided
        if V0_init is not None:
            V0_guess = V0_init
        if B0_init is not None:
            B0_guess = B0_init
        if B0_prime_init is not None:
            B0_prime_guess = B0_prime_init

        B0_prime2_guess = -0.02

        # Set bounds
        V0_min = np.max(V_data) * self.V0_bounds[0]
        V0_max = np.max(V_data) * self.V0_bounds[1]
        B0_min, B0_max = self.B0_bounds
        B0p_min, B0p_max = self.B0_prime_bounds

        # Respect locked parameters by collapsing their bounds to a single value
        if lock_flags.get('V0'):
            V0_min = V0_max = V0_guess
        if lock_flags.get('B0'):
            B0_min = B0_max = B0_guess
        if lock_flags.get('B0_prime'):
            B0p_min = B0p_max = B0_prime_guess

        # Simple equal weighting for nonlinear fit
        weights = np.ones_like(P_data)

        try:
            # Determine parameters and bounds
            if self.eos_type == EoSType.BIRCH_MURNAGHAN_2ND:
                # 2 parameters: V0, B0
                bounds = ([V0_min, B0_min],
                         [V0_max, B0_max])
                p0 = [V0_guess, B0_guess]
                n_params = 2

            elif self.eos_type == EoSType.BIRCH_MURNAGHAN_4TH:
                # 4 parameters: V0, B0, B0', B0''
                bounds = ([V0_min, B0_min, B0p_min, -0.1],
                         [V0_max, B0_max, B0p_max, 0.0])
                p0 = [V0_guess, B0_guess, B0_prime_guess, B0_prime2_guess]
                n_params = 4

            elif self.eos_type == EoSType.TAIT:
                # Can use 3 or 4 parameters
                bounds = ([V0_min, B0_min, B0p_min],
                         [V0_max, B0_max, B0p_max])
                p0 = [V0_guess, B0_guess, B0_prime_guess]
                n_params = 3

            else:
                # 3 parameters: V0, B0, B0' (default for most EoS)
                bounds = ([V0_min, B0_min, B0p_min],
                         [V0_max, B0_max, B0p_max])
                p0 = [V0_guess, B0_guess, B0_prime_guess]
                n_params = 3

            # Perform weighted curve fitting with balanced settings
            # Priority: fit quality first, then stability
            popt, pcov = curve_fit(
                self.eos_function,
                V_data,
                P_data,
                p0=p0,
                bounds=bounds,
                sigma=1.0/weights,
                absolute_sigma=False,
                method='trf',  # Trust Region Reflective
                maxfev=self.max_iterations,
                ftol=1e-10,  # Reasonable tolerance for good convergence
                xtol=1e-10
            )

            # Extract parameters
            perr = np.sqrt(np.diag(pcov))

            # Calculate fitted values and statistics
            P_fit = self.eos_function(V_data, *popt)
            residuals = P_data - P_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((P_data - np.mean(P_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))

            # Calculate reduced chi-square
            dof = len(V_data) - len(popt)
            if dof > 0:
                # Weighted residuals (using 1/weights as variance)
                chi2_reduced = ss_res / dof
            else:
                chi2_reduced = 0.0

            # Create parameter object
            params = EoSParameters(eos_type=self.eos_type)
            params.V0 = popt[0]
            params.V0_err = perr[0]
            params.B0 = popt[1]
            params.B0_err = perr[1]

            if len(popt) >= 3:
                params.B0_prime = popt[2]
                params.B0_prime_err = perr[2]
            else:
                params.B0_prime = 4.0  # Fixed for 2nd order BM
                params.B0_prime_err = 0.0

            if len(popt) >= 4:
                params.B0_prime2 = popt[3]
                params.B0_prime2_err = perr[3]

            params.R_squared = r_squared
            params.RMSE = rmse
            params.chi2 = chi2_reduced
            params.n_data = len(V_data)

            return params

        except Exception as e:
            print(f"Fitting failed for {self.eos_type.value}: {e}")
            return None

    def refine_with_locked(self, V_data: np.ndarray, P_data: np.ndarray,
                           base_params: EoSParameters,
                           lock_flags: Dict[str, bool],
                           step_fraction: float = 0.05) -> Optional[EoSParameters]:
        """
        Light-weight refinement that only adjusts unlocked parameters.

        The search window around the current values is intentionally small
        (``step_fraction`` of the current value) to prevent the optimizer from
        wandering far away when the user intends to tweak a single parameter.
        """
        V_data = np.array(V_data)
        P_data = np.array(P_data)

        # Build lists of free parameters
        names = ['V0', 'B0', 'B0_prime']
        current = [base_params.V0, base_params.B0, base_params.B0_prime]
        base_b0_prime2 = getattr(base_params, 'B0_prime2', 0.0)
        free_indices = [i for i, n in enumerate(names) if not lock_flags.get(n, False)]

        if not free_indices:
            return base_params

        x0 = np.array([current[i] for i in free_indices], dtype=float)
        lower_bounds = []
        upper_bounds = []

        for name, value in zip([names[i] for i in free_indices], x0):
            delta = max(abs(value) * step_fraction, 1e-6)

            if name == 'V0':
                lower = max(value - max(delta, 1e-3), 1e-6)
                upper = value + max(delta, 1e-3)
            elif name == 'B0':
                lower = max(self.B0_bounds[0], value - max(delta, 0.5))
                upper = min(self.B0_bounds[1], value + max(delta, 0.5))
            else:  # B0_prime
                lower = max(self.B0_prime_bounds[0], value - max(delta, 0.2))
                upper = min(self.B0_prime_bounds[1], value + max(delta, 0.2))

            lower_bounds.append(lower)
            upper_bounds.append(upper)

        def residuals(x_vector: np.ndarray) -> np.ndarray:
            full = current[:]
            for idx, free_idx in enumerate(free_indices):
                full[free_idx] = x_vector[idx]

            params = EoSParameters(eos_type=self.eos_type,
                                   V0=full[0], B0=full[1], B0_prime=full[2],
                                   B0_prime2=base_b0_prime2)
            return self.calculate_pressure(V_data, params) - P_data

        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                ftol=1e-12,
                xtol=1e-12,
                max_nfev=2000,
            )
        except Exception:
            return None

        full = current[:]
        for idx, free_idx in enumerate(free_indices):
            full[free_idx] = result.x[idx]

        params = EoSParameters(eos_type=self.eos_type)
        params.V0 = full[0]
        params.B0 = full[1]
        params.B0_prime = full[2]
        params.B0_prime2 = base_b0_prime2

        P_fit = self.calculate_pressure(V_data, params)
        residuals_p = P_data - P_fit
        ss_res = np.sum(residuals_p**2)
        ss_tot = np.sum((P_data - np.mean(P_data))**2)
        params.R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        params.RMSE = np.sqrt(np.mean(residuals_p**2))
        params.n_data = len(V_data)

        return params

    def calculate_pressure(self, V: np.ndarray, params: EoSParameters) -> np.ndarray:
        """
        Calculate pressure from volume using fitted parameters

        Parameters:
        -----------
        V : array
            Volume (Å³/atom)
        params : EoSParameters
            Fitted EoS parameters

        Returns:
        --------
        P : array
            Pressure (GPa)
        """
        if self.eos_type == EoSType.BIRCH_MURNAGHAN_2ND:
            return self.eos_function(V, params.V0, params.B0)
        elif self.eos_type == EoSType.BIRCH_MURNAGHAN_4TH:
            return self.eos_function(V, params.V0, params.B0, params.B0_prime, params.B0_prime2)
        else:
            return self.eos_function(V, params.V0, params.B0, params.B0_prime)

    def fit_with_multiple_strategies(self, V_data: np.ndarray, P_data: np.ndarray,
                                    verbose: bool = False) -> EoSParameters:
        """
        Try multiple fitting strategies to find the best fit
        
        Flexible approach: tries different initial guesses
        Selection based on fit quality (R² and RMSE)
        
        Parameters:
        -----------
        V_data : array
            Volume data (Å³/atom)
        P_data : array
            Pressure data (GPa)
        verbose : bool
            Print progress messages (default: False)
            
        Returns:
        --------
        params : EoSParameters
            Best fitted parameters, or None if all strategies fail
        """
        if verbose:
            print("\nTrying multiple fitting strategies...")
        
        strategies = []
        
        # Strategy 1: Smart initial guess (default)
        if verbose:
            print("  Strategy 1: Smart initial guess...")
        try:
            params = self.fit(V_data, P_data, use_smart_guess=True)
            if params is not None:
                strategies.append(('Smart Guess', params))
                if verbose:
                    print(f"    Success: R2 = {params.R_squared:.6f}, RMSE = {params.RMSE:.4f}, B0' = {params.B0_prime:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
        
        # Strategy 2: Simple guess
        if verbose:
            print("  Strategy 2: Simple initial guess...")
        try:
            params = self.fit(V_data, P_data, use_smart_guess=False)
            if params is not None:
                strategies.append(('Simple Guess', params))
                if verbose:
                    print(f"    Success: R2 = {params.R_squared:.6f}, RMSE = {params.RMSE:.4f}, B0' = {params.B0_prime:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
        
        # Strategy 3: Different B0_prime starting values
        for B0p_start in [3.5, 4.0, 4.5]:
            if verbose:
                print(f"  Strategy: B0_prime start = {B0p_start}...")
            try:
                V0_guess = np.max(V_data) * 1.03
                B0_guess = 150.0
                params = self.fit(V_data, P_data, use_smart_guess=False,
                                V0_init=V0_guess, B0_init=B0_guess, B0_prime_init=B0p_start)
                if params is not None:
                    strategies.append((f'B0p={B0p_start}', params))
                    if verbose:
                        print(f"    Success: R2 = {params.R_squared:.6f}, RMSE = {params.RMSE:.4f}, B0' = {params.B0_prime:.3f}")
            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")
        
        # Select best strategy - prioritize R² but also check RMSE
        if not strategies:
            if verbose:
                print("\nAll strategies failed!")
            return None
        
        # Sort by R² first, then by RMSE if R² is similar
        strategies.sort(key=lambda x: (-x[1].R_squared, x[1].RMSE))
        best_strategy, best_params = strategies[0]
        
        if verbose:
            print(f"\nBest strategy: {best_strategy}")
            print(f"  V0 = {best_params.V0:.4f}")
            print(f"  B0 = {best_params.B0:.2f} GPa")
            print(f"  B0' = {best_params.B0_prime:.3f}")
            print(f"  R2 = {best_params.R_squared:.6f}")
            print(f"  RMSE = {best_params.RMSE:.4f} GPa")
        
        return best_params

    def print_parameters(self, params: EoSParameters, phase_name: str = ""):
        """
        Print fitted parameters in a formatted way (CrysFML style)

        Parameters:
        -----------
        params : EoSParameters
            Fitted parameters
        phase_name : str
            Name of the phase for display
        """
        print(f"\n{'='*70}")
        if phase_name:
            print(f"{phase_name} - {params.eos_type.value.upper()} EoS Fitting Results:")
        else:
            print(f"{params.eos_type.value.upper()} EoS Fitting Results:")
        print(f"{'='*70}")

        # Parameters with uncertainties
        print("\nFitted Parameters:")
        print(f"  V₀      = {params.V0:.5f} ± {params.V0_err:.5f} Å³/atom")
        print(f"  B₀      = {params.B0:.3f} ± {params.B0_err:.3f} GPa")
        print(f"  B₀'     = {params.B0_prime:.4f} ± {params.B0_prime_err:.4f}")

        # Warning if B0' is at boundary (indicates potential divergence)
        if params.eos_type == EoSType.BIRCH_MURNAGHAN_3RD:
            if abs(params.B0_prime - 2.0) < 0.01 or abs(params.B0_prime - 8.0) < 0.01:
                print("\n  ⚠️  WARNING: B0' is at physical boundary!")
                print("      This may indicate:")
                print("      1. BM3 model is not suitable for this data")
                print("      2. Consider using Vinet or other EoS models")
                print(f"      3. Try increasing regularization_strength (current: {self.regularization_strength:.1f})")
                print("      4. Check data quality and pressure range")

        if abs(params.B0_prime2) > 1e-10:
            print(f"  B₀''    = {params.B0_prime2:.6f} ± {params.B0_prime2_err:.6f} GPa⁻¹")

        # Quality metrics (following CrysFML/EosFit7c format)
        print("\nFit Quality:")
        print(f"  N data  = {params.n_data}")
        print(f"  R²      = {params.R_squared:.6f}")
        print(f"  RMSE    = {params.RMSE:.4f} GPa")
        print(f"  χ²ᵣ     = {params.chi2:.4f}")

        # Fit quality assessment
        if params.chi2 > 0:
            if params.chi2 < 1.5:
                quality = "Excellent"
            elif params.chi2 < 3.0:
                quality = "Good"
            elif params.chi2 < 5.0:
                quality = "Acceptable"
            else:
                quality = "Poor - check data quality"
            print(f"  Quality: {quality}")

        print(f"{'='*70}\n")


class MultiEoSFitter:
    """
    Multi-EoS Comparison Tool

    Fits multiple EoS models to the same data and compares results
    """

    def __init__(self, V_data: np.ndarray, P_data: np.ndarray):
        """
        Initialize with P-V data

        Parameters:
        -----------
        V_data : array
            Volume data (Å³/atom)
        P_data : array
            Pressure data (GPa)
        """
        self.V_data = np.array(V_data)
        self.P_data = np.array(P_data)
        self.results = {}

    def fit_all_models(self, eos_types: Optional[List[EoSType]] = None) -> Dict[EoSType, EoSParameters]:
        """
        Fit all specified EoS models

        Parameters:
        -----------
        eos_types : list of EoSType, optional
            List of EoS types to fit. If None, fits all available models.

        Returns:
        --------
        results : dict
            Dictionary mapping EoS types to fitted parameters
        """
        if eos_types is None:
            eos_types = [
                EoSType.MURNAGHAN,
                EoSType.BIRCH_MURNAGHAN_2ND,
                EoSType.BIRCH_MURNAGHAN_3RD,
                EoSType.VINET,
                EoSType.NATURAL_STRAIN
            ]

        print("\n🔧 Fitting multiple EoS models...")

        for eos_type in eos_types:
            fitter = CrysFMLEoS(eos_type=eos_type)
            params = fitter.fit(self.V_data, self.P_data)

            if params is not None:
                self.results[eos_type] = params
                fitter.print_parameters(params)

        return self.results

    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison table of all fitted models

        Returns:
        --------
        df : DataFrame
            Comparison table
        """
        if not self.results:
            print("❌ No results available. Run fit_all_models() first.")
            return None

        comparison_data = []

        for eos_type, params in self.results.items():
            row = {
                'EoS Model': eos_type.value,
                'V₀ (Å³/atom)': f"{params.V0:.4f} ± {params.V0_err:.4f}",
                'B₀ (GPa)': f"{params.B0:.2f} ± {params.B0_err:.2f}",
                "B₀'": f"{params.B0_prime:.3f} ± {params.B0_prime_err:.3f}",
                'R²': f"{params.R_squared:.6f}",
                'RMSE (GPa)': f"{params.RMSE:.4f}"
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of all fitted models

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        if not self.results:
            print("❌ No results available. Run fit_all_models() first.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data
        ax.scatter(self.V_data, self.P_data, s=100, c='black', marker='o',
                  label='Experimental Data', alpha=0.7, edgecolors='black', zorder=5)

        # Plot fitted curves
        V_fit = np.linspace(self.V_data.min()*0.95, self.V_data.max()*1.05, 300)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

        for idx, (eos_type, params) in enumerate(self.results.items()):
            fitter = CrysFMLEoS(eos_type=eos_type)
            P_fit = fitter.calculate_pressure(V_fit, params)

            label = f"{eos_type.value} (R²={params.R_squared:.4f})"
            ax.plot(V_fit, P_fit, color=colors[idx % len(colors)],
                   linewidth=2, label=label, alpha=0.8)

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=13)
        ax.set_ylabel('Pressure P (GPa)', fontsize=13)
        ax.set_title('Comparison of Multiple EoS Models', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Comparison plot saved to: {save_path}")

        plt.show()


class InteractiveEoSFitter:
    """
    Interactive EoS Fitter with Manual Parameter Adjustment

    Allows manual adjustment of V0, B0, and B0' parameters with real-time
    visualization of fitting quality and residuals.
    """

    def __init__(self, V_data: np.ndarray, P_data: np.ndarray,
                 eos_type: EoSType = EoSType.BIRCH_MURNAGHAN_3RD):
        """
        Initialize Interactive EoS Fitter

        Parameters:
        -----------
        V_data : array
            Volume data (Å³/atom)
        P_data : array
            Pressure data (GPa)
        eos_type : EoSType
            Type of equation of state to use
        """
        self.V_data = np.array(V_data)
        self.P_data = np.array(P_data)
        self.eos_type = eos_type
        self.fitter = CrysFMLEoS(eos_type=eos_type)

        # Current parameters (will be set by auto-fit or manual adjustment)
        self.current_params = None
        self.auto_fit_params = None

    def auto_fit(self, verbose: bool = True) -> EoSParameters:
        """
        Perform automatic fitting

        Parameters:
        -----------
        verbose : bool
            Print fitting results (default: True)

        Returns:
        --------
        params : EoSParameters
            Fitted parameters
        """
        self.auto_fit_params = self.fitter.fit(self.V_data, self.P_data)
        self.current_params = self.auto_fit_params

        if verbose and self.auto_fit_params is not None:
            self.fitter.print_parameters(self.auto_fit_params, "Auto-Fit Results")

        return self.auto_fit_params

    def manual_fit(self, V0: float, B0: float, B0_prime: float,
                   verbose: bool = True) -> EoSParameters:
        """
        Create parameter set from manual values

        Parameters:
        -----------
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            Pressure derivative of bulk modulus
        verbose : bool
            Print results (default: True)

        Returns:
        --------
        params : EoSParameters
            Parameter set with manual values
        """
        # Calculate fitted P values
        P_fit = self.fitter.eos_function(self.V_data, V0, B0, B0_prime)

        # Calculate statistics
        residuals = self.P_data - P_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.P_data - np.mean(self.P_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))

        # Create parameter object
        params = EoSParameters(eos_type=self.eos_type)
        params.V0 = V0
        params.B0 = B0
        params.B0_prime = B0_prime
        params.R_squared = r_squared
        params.RMSE = rmse

        self.current_params = params

        if verbose:
            self.fitter.print_parameters(params, "Manual Parameters")

        return params

    def refine_from_manual(self, V0: float, B0: float, B0_prime: float,
                          verbose: bool = True) -> EoSParameters:
        """
        Refine fit starting from manual parameter values

        Parameters:
        -----------
        V0 : float
            Initial V0 value
        B0 : float
            Initial B0 value
        B0_prime : float
            Initial B0' value
        verbose : bool
            Print results (default: True)

        Returns:
        --------
        params : EoSParameters
            Refined fitted parameters
        """
        refined_params = self.fitter.fit(
            self.V_data, self.P_data,
            use_smart_guess=False,
            V0_init=V0,
            B0_init=B0,
            B0_prime_init=B0_prime
        )

        self.current_params = refined_params

        if verbose and refined_params is not None:
            self.fitter.print_parameters(refined_params, "Refined Fit Results")

        return refined_params

    def plot_fit_with_residuals(self, params: Optional[EoSParameters] = None,
                                show_auto_fit: bool = True,
                                save_path: Optional[str] = None):
        """
        Plot fitting results with residual analysis

        Parameters:
        -----------
        params : EoSParameters, optional
            Parameters to plot. If None, uses current_params
        show_auto_fit : bool
            Also show auto-fit results for comparison (default: True)
        save_path : str, optional
            Path to save the figure
        """
        if params is None:
            params = self.current_params

        if params is None:
            print("❌ No parameters available. Run auto_fit() or manual_fit() first.")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # Main P-V plot
        ax_main = fig.add_subplot(gs[0:2, 0])

        # Plot experimental data
        ax_main.scatter(self.V_data, self.P_data, s=120, c='black', marker='o',
                       label='Experimental Data', alpha=0.8, edgecolors='black',
                       linewidths=1.5, zorder=5)

        # Plot current fit
        V_fit = np.linspace(self.V_data.min()*0.95, self.V_data.max()*1.05, 300)
        P_fit = self.fitter.calculate_pressure(V_fit, params)
        ax_main.plot(V_fit, P_fit, color='red', linewidth=2.5,
                    label=f'Current Fit (R²={params.R_squared:.4f})', alpha=0.9)

        # Plot auto-fit if requested and available
        if show_auto_fit and self.auto_fit_params is not None and params != self.auto_fit_params:
            P_auto = self.fitter.calculate_pressure(V_fit, self.auto_fit_params)
            ax_main.plot(V_fit, P_auto, color='blue', linewidth=2, linestyle='--',
                        label=f'Auto-Fit (R²={self.auto_fit_params.R_squared:.4f})',
                        alpha=0.7)

        ax_main.set_xlabel('Volume V (Å³/atom)', fontsize=13, fontweight='bold')
        ax_main.set_ylabel('Pressure P (GPa)', fontsize=13, fontweight='bold')
        ax_main.set_title(f'{self.eos_type.value.upper()} Equation of State Fitting',
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle='--')

        # Parameter box
        textstr = f"V₀ = {params.V0:.4f} Å³/atom\n"
        textstr += f"B₀ = {params.B0:.2f} GPa\n"
        textstr += f"B₀' = {params.B0_prime:.3f}\n"
        textstr += f"R² = {params.R_squared:.6f}\n"
        textstr += f"RMSE = {params.RMSE:.4f} GPa"
        ax_main.text(0.05, 0.95, textstr, transform=ax_main.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Residuals plot
        ax_res = fig.add_subplot(gs[2, 0])
        P_data_fit = self.fitter.calculate_pressure(self.V_data, params)
        residuals = self.P_data - P_data_fit

        ax_res.scatter(self.V_data, residuals, s=80, c='red', marker='o',
                      alpha=0.7, edgecolors='black', linewidths=1)
        ax_res.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax_res.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax_res.set_ylabel('Residuals (GPa)', fontsize=12)
        ax_res.set_title('Fitting Residuals', fontsize=13, fontweight='bold')
        ax_res.grid(True, alpha=0.3)

        # Residual histogram
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_hist.hist(residuals, bins=min(15, len(residuals)), color='skyblue',
                    edgecolor='black', alpha=0.7)
        ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax_hist.set_xlabel('Residual (GPa)', fontsize=12)
        ax_hist.set_ylabel('Frequency', fontsize=12)
        ax_hist.set_title('Residual Distribution', fontsize=13, fontweight='bold')
        ax_hist.grid(True, alpha=0.3, axis='y')

        # Statistics text
        stats_text = f"Residual Statistics:\n"
        stats_text += f"Mean: {np.mean(residuals):.4f} GPa\n"
        stats_text += f"Std Dev: {np.std(residuals):.4f} GPa\n"
        stats_text += f"Max: {np.max(np.abs(residuals)):.4f} GPa"
        ax_hist.text(0.98, 0.97, stats_text, transform=ax_hist.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Relative error plot
        ax_rel = fig.add_subplot(gs[1, 1])
        # Avoid division by very small pressures
        rel_error = np.where(np.abs(self.P_data) > 0.1,
                            (residuals / self.P_data) * 100, 0)
        ax_rel.scatter(self.P_data, rel_error, s=80, c='green', marker='s',
                      alpha=0.7, edgecolors='black', linewidths=1)
        ax_rel.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax_rel.set_xlabel('Pressure P (GPa)', fontsize=12)
        ax_rel.set_ylabel('Relative Error (%)', fontsize=12)
        ax_rel.set_title('Relative Fitting Error', fontsize=13, fontweight='bold')
        ax_rel.grid(True, alpha=0.3)

        # Data quality table
        ax_table = fig.add_subplot(gs[2, 1])
        ax_table.axis('off')

        table_data = []
        table_data.append(['Metric', 'Value'])
        table_data.append(['Data Points', f'{len(self.V_data)}'])
        table_data.append(['R²', f'{params.R_squared:.6f}'])
        table_data.append(['RMSE', f'{params.RMSE:.4f} GPa'])
        table_data.append(['Max |Residual|', f'{np.max(np.abs(residuals)):.4f} GPa'])
        table_data.append(['Mean |Residual|', f'{np.mean(np.abs(residuals)):.4f} GPa'])

        table = ax_table.table(cellText=table_data, cellLoc='left',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        fig.suptitle(f'Interactive EoS Fitting Analysis - {self.eos_type.value.upper()}',
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Figure saved to: {save_path}")

        plt.show()

    def compare_parameter_sets(self, param_sets: Dict[str, EoSParameters],
                              save_path: Optional[str] = None):
        """
        Compare multiple parameter sets visually

        Parameters:
        -----------
        param_sets : dict
            Dictionary mapping labels to EoSParameters objects
        save_path : str, optional
            Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot experimental data
        ax1.scatter(self.V_data, self.P_data, s=120, c='black', marker='o',
                   label='Experimental Data', alpha=0.8, edgecolors='black',
                   linewidths=1.5, zorder=5)

        # Plot fits
        V_fit = np.linspace(self.V_data.min()*0.95, self.V_data.max()*1.05, 300)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        residuals_dict = {}
        for idx, (label, params) in enumerate(param_sets.items()):
            P_fit = self.fitter.calculate_pressure(V_fit, params)
            color = colors[idx % len(colors)]

            ax1.plot(V_fit, P_fit, color=color, linewidth=2.5,
                    label=f'{label} (R²={params.R_squared:.4f})', alpha=0.8)

            # Calculate residuals for this parameter set
            P_data_fit = self.fitter.calculate_pressure(self.V_data, params)
            residuals_dict[label] = (self.P_data - P_data_fit, color)

        ax1.set_xlabel('Volume V (Å³/atom)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Pressure P (GPa)', fontsize=13, fontweight='bold')
        ax1.set_title('P-V Curves Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Plot residuals comparison
        for label, (residuals, color) in residuals_dict.items():
            ax2.scatter(self.V_data, residuals, s=80, marker='o',
                       label=label, alpha=0.7, edgecolors='black',
                       linewidths=1, color=color)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax2.set_xlabel('Volume V (Å³/atom)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Residuals (GPa)', fontsize=13, fontweight='bold')
        ax2.set_title('Residuals Comparison', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Comparison figure saved to: {save_path}")

        plt.show()

    def get_comparison_table(self, param_sets: Dict[str, EoSParameters]) -> pd.DataFrame:
        """
        Create comparison table for multiple parameter sets

        Parameters:
        -----------
        param_sets : dict
            Dictionary mapping labels to EoSParameters objects

        Returns:
        --------
        df : DataFrame
            Comparison table
        """
        comparison_data = []

        for label, params in param_sets.items():
            row = {
                'Parameter Set': label,
                'V₀ (Å³/atom)': f"{params.V0:.4f}",
                'B₀ (GPa)': f"{params.B0:.2f}",
                "B₀'": f"{params.B0_prime:.3f}",
                'R²': f"{params.R_squared:.6f}",
                'RMSE (GPa)': f"{params.RMSE:.4f}"
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example data (replace with real data)
    V_data = np.array([17.5, 17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0])
    P_data = np.array([0.0, 5.2, 10.8, 17.1, 24.2, 32.1, 41.0, 51.0])

    print("\n" + "="*80)
    print("CrysFML EoS Module - Enhanced Interactive Fitting Example")
    print("="*80)

    # ==================== Method 1: Basic Fitting ====================
    print("\n" + "="*80)
    print("📊 Method 1: Basic 3rd Order Birch-Murnaghan Fitting")
    print("="*80)

    bm3_fitter = CrysFMLEoS(eos_type=EoSType.BIRCH_MURNAGHAN_3RD)
    bm3_params = bm3_fitter.fit(V_data, P_data)
    bm3_fitter.print_parameters(bm3_params, "Example Phase")

    # ==================== Method 2: Interactive Fitting ====================
    print("\n" + "="*80)
    print("📊 Method 2: Interactive Fitting with Manual Adjustment")
    print("="*80)

    # Create interactive fitter
    interactive_fitter = InteractiveEoSFitter(V_data, P_data,
                                             eos_type=EoSType.BIRCH_MURNAGHAN_3RD)

    # Auto-fit first
    print("\n🔧 Step 1: Auto-fit...")
    auto_params = interactive_fitter.auto_fit()

    # Manually adjust parameters (example: slightly modify V0 and B0)
    print("\n🔧 Step 2: Manual adjustment...")
    manual_params = interactive_fitter.manual_fit(
        V0=17.8,    # Manually set V0
        B0=140.0,   # Manually set B0
        B0_prime=4.2  # Manually set B0'
    )

    # Refine from manual values
    print("\n🔧 Step 3: Refine from manual values...")
    refined_params = interactive_fitter.refine_from_manual(
        V0=17.8,
        B0=140.0,
        B0_prime=4.2
    )

    # Plot comprehensive analysis with residuals
    print("\n📈 Plotting comprehensive analysis...")
    interactive_fitter.plot_fit_with_residuals(refined_params, show_auto_fit=True)

    # Compare all three parameter sets
    print("\n📊 Comparing auto-fit, manual, and refined parameters...")
    param_sets = {
        'Auto-Fit': auto_params,
        'Manual': manual_params,
        'Refined': refined_params
    }

    # Show comparison table
    comparison_df = interactive_fitter.get_comparison_table(param_sets)
    print("\n📋 Parameter Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Plot comparison
    interactive_fitter.compare_parameter_sets(param_sets)

    # ==================== Method 3: Multi-Model Comparison ====================
    print("\n" + "="*80)
    print("📊 Method 3: Comparing Multiple EoS Models")
    print("="*80)

    multi_fitter = MultiEoSFitter(V_data, P_data)
    results = multi_fitter.fit_all_models()

    # Show comparison table
    print("\n📋 Multi-Model Comparison Table:")
    comparison_df = multi_fitter.compare_models()
    print(comparison_df.to_string(index=False))

    # Plot comparison
    multi_fitter.plot_comparison()

    print("\n" + "="*80)
    print("✨ Interactive EoS Fitting Complete!")
    print("="*80)
    print("\n💡 Key Features:")
    print("   1. Smart initial guess for better convergence")
    print("   2. Manual parameter adjustment (V0, B0, B0')")
    print("   3. Real-time visualization with residuals")
    print("   4. Multiple parameter set comparison")
    print("   5. Comprehensive fitting quality metrics")
    print("="*80)