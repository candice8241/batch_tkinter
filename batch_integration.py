#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced script for batch processing HDF5 diffraction images - Enhanced Version
Performs 1D integration using pyFAI with multiple output formats and stacked plotting

Features:
- Multiple output formats: .xy, .dat, .chi, .svg, .png, .fxye
- Automatic stacked pressure plot generation
- Pressure extraction from filenames
- Color-coded by pressure range (changes every 10 GPa)

Usage:
    python batch_integration_advanced.py                    # Use default config file
    python batch_integration_advanced.py config.ini         # Use specified config file
    python batch_integration_advanced.py --help             # Show help information

Author: Felicity 💕
"""

import os
import sys
import glob
import h5py
import numpy as np
import pyFAI
import fabio
import argparse
import configparser
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
# Fix Tcl_AsyncDelete threading error: Set non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter thread conflicts
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class BatchIntegrator:
    """Batch integration processor"""
    
    def __init__(self, poni_file, mask_file=None):
        """
        Initialize the integrator
        
        Args:
            poni_file (str): Path to calibration file (.poni)
            mask_file (str, optional): Path to mask file
        """
        self.ai = pyFAI.load(poni_file)
        print(f"✓ Successfully loaded calibration file: {poni_file}")
        print(f"  Detector: {self.ai.detector}")
        print(f"  Wavelength: {self.ai.wavelength*1e10:.4f} Å")
        print(f"  Sample-detector distance: {self.ai.dist*1000:.2f} mm")
        
        self.mask = None
        if mask_file and os.path.exists(mask_file):
            self.mask = self._load_mask(mask_file)
            print(f"✓ Successfully loaded mask file: {mask_file}")
            print(f"  Mask shape: {self.mask.shape}")
            print(f"  Masked pixels: {np.sum(self.mask)}")
        elif mask_file:
            print(f"⚠ Warning: Mask file not found: {mask_file}")
    
    def _load_mask(self, mask_file):
        """Load mask file"""
        ext = os.path.splitext(mask_file)[1].lower()
        
        if ext == '.npy':
            mask = np.load(mask_file)
        elif ext in ['.edf', '.tif', '.tiff', '.png']:
            mask = fabio.open(mask_file).data
        else:
            raise ValueError(f"Unsupported mask file format: {ext}")
        
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        return mask
    
    def _read_h5_image(self, h5_file, dataset_path=None, frame_index=0):
        """
        Read image data from HDF5 file
        
        Args:
            h5_file (str): Path to HDF5 file
            dataset_path (str, optional): Dataset path within HDF5
            frame_index (int): Frame index if multi-frame data
        
        Returns:
            numpy.ndarray: Image data
        """
        with h5py.File(h5_file, 'r') as f:
            if dataset_path is None:
                dataset_path = self._find_image_dataset(f)
            
            if dataset_path not in f:
                raise ValueError(f"Dataset not found in HDF5 file: {dataset_path}")
            
            data = f[dataset_path]
            
            if len(data.shape) == 3:
                if frame_index >= data.shape[0]:
                    raise ValueError(f"Frame index {frame_index} out of bounds (total frames: {data.shape[0]})")
                img_data = data[frame_index]
            else:
                img_data = data[()]
        
        return img_data
    
    def _find_image_dataset(self, h5_file_obj):
        """Automatically find image dataset path in HDF5"""
        common_paths = [
            '/entry/data/data',
            '/entry/instrument/detector/data',
            '/entry/data/image',
            '/data/data',
            '/data',
            '/image',
            'data',
        ]
        
        for path in common_paths:
            if path in h5_file_obj:
                return path
        
        def find_dataset(obj, path=''):
            if isinstance(obj, h5py.Dataset):
                if len(obj.shape) >= 2:
                    return path
            elif isinstance(obj, h5py.Group):
                for key in obj.keys():
                    result = find_dataset(obj[key], path + '/' + key)
                    if result:
                        return result
            return None
        
        result = find_dataset(h5_file_obj)
        if result:
            print(f"  Automatically found dataset: {result}")
            return result
        else:
            raise ValueError("No suitable image dataset found in HDF5 file")
    
    def integrate_single(self, h5_file, output_base, npt=2000, unit="2th_deg",
                        dataset_path=None, frame_index=0, formats=['xy'], **kwargs):
        """
        Integrate a single HDF5 file and save in multiple formats

        Args:
            h5_file (str): Input HDF5 file
            output_base (str): Output file base path (without extension)
            npt (int): Number of points for integration
            unit (str): Output unit
            dataset_path (str, optional): Dataset path
            frame_index (int): Frame index (for multi-frame)
            formats (list): List of output formats ['xy', 'dat', 'chi', 'fxye']
            **kwargs: Additional arguments to integrate1d
        """
        try:
            img_data = self._read_h5_image(h5_file, dataset_path, frame_index)

            # Perform integration
            result = self.ai.integrate1d(
                img_data,
                npt=npt,
                mask=self.mask,
                unit=unit,
                **kwargs
            )

            # Save in multiple formats
            for fmt in formats:
                output_file = f"{output_base}.{fmt}"

                if fmt == 'xy':
                    self._save_xy(result, output_file)
                elif fmt == 'dat':
                    self._save_dat(result, output_file)
                elif fmt == 'chi':
                    self._save_chi(result, output_file)
                elif fmt == 'fxye':
                    self._save_fxye(result, output_file)
                elif fmt == 'svg':
                    self._save_svg(result, output_file)
                elif fmt == 'png':
                    self._save_png(result, output_file)

            return True, None

        except Exception as e:
            return False, str(e)

    def _save_xy(self, result, filename):
        """Save result in .xy format"""
        np.savetxt(filename, np.column_stack(result), fmt='%.6f')

    def _save_dat(self, result, filename):
        """Save result in .dat format (same as .xy)"""
        np.savetxt(filename, np.column_stack(result), fmt='%.6f')

    def _save_chi(self, result, filename):
        """Save result in .chi format (GSAS-II compatible)"""
        with open(filename, 'w') as f:
            f.write(f"# Chi file generated by pyFAI\n")
            f.write(f"# 2theta (deg) Intensity\n")
            for x, y in zip(result[0], result[1]):
                f.write(f"{x:12.6f} {y:16.6f}\n")

    def _save_fxye(self, result, filename):
        """Save result in .fxye format (GSAS compatible)"""
        with open(filename, 'w') as f:
            f.write("TITLE pyFAI integration\n")
            f.write(f"BANK 1 {len(result[0])} 1 CONS {result[0][0]:.6f} {(result[0][1]-result[0][0]):.6f} 0 0 FXYE\n")
            for x, y in zip(result[0], result[1]):
                esd = np.sqrt(y) if y > 0 else 1.0
                f.write(f"{x:15.6f} {y:15.6f} {esd:15.6f}\n")

    def _save_svg(self, result, filename):
        """Save result as SVG plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(result[0], result[1], 'b-', linewidth=1)
        plt.xlabel('2θ (deg)' if '2th' in str(result) else 'Q (Å⁻¹)')
        plt.ylabel('Intensity')
        plt.title('Integrated Diffraction Pattern')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, format='svg')
        plt.close()

    def _save_png(self, result, filename):
        """Save result as PNG plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(result[0], result[1], 'b-', linewidth=1)
        plt.xlabel('2θ (deg)' if '2th' in str(result) else 'Q (Å⁻¹)')
        plt.ylabel('Intensity')
        plt.title('Integrated Diffraction Pattern')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, format='png', dpi=300)
        plt.close()
    
    def batch_integrate(self, input_pattern, output_dir, npt=2000, unit="2th_deg",
                        dataset_path=None, formats=['xy'], create_stacked_plot=False,
                        stacked_plot_offset='auto', **kwargs):
        """
        Batch integration for multiple HDF5 files

        Args:
            formats (list): Output formats ['xy', 'dat', 'chi', 'svg', 'png', 'fxye']
            create_stacked_plot (bool): Whether to create stacked plot
            stacked_plot_offset (str or float): Offset for stacked plot ('auto' or float value)
        """
        # Enhanced file search with multiple attempts
        h5_files = sorted(glob.glob(input_pattern, recursive=True))

        # If no files found and pattern doesn't use **, try adding ** for recursive search
        if not h5_files and '**' not in input_pattern:
            # Try to construct a recursive pattern
            if input_pattern.endswith('*.h5'):
                # Replace *.h5 with **/*.h5 for recursive search
                recursive_pattern = input_pattern.replace('*.h5', '**/*.h5')
                h5_files = sorted(glob.glob(recursive_pattern, recursive=True))
                if h5_files:
                    print(f"ℹ Using recursive search pattern: {recursive_pattern}")

        if not h5_files:
            print(f"⚠ No matching files found: {input_pattern}")
            print(f"  Tip: Use '**/*.h5' pattern for recursive directory search")
            return

        print(f"\nFound {len(h5_files)} HDF5 files to process")
        print(f"Output directory: {output_dir}")
        print(f"Integration parameters: {npt} points, unit={unit}")
        print(f"Output formats: {', '.join(formats)}\n")

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        failed_files = []

        for h5_file in tqdm(h5_files, desc="Processing"):
            basename = os.path.splitext(os.path.basename(h5_file))[0]
            output_base = os.path.join(output_dir, basename)

            success, error_msg = self.integrate_single(
                h5_file, output_base, npt, unit, dataset_path, formats=formats, **kwargs
            )

            if success:
                success_count += 1
                print(f"✓ Success: {h5_file} -> {output_base}.[{','.join(formats)}]")
            else:
                failed_files.append((h5_file, error_msg))
                print(f"✗ Failed: {h5_file}\n  Error: {error_msg}")

        print(f"\n✓ Batch processing complete!")
        print(f"  Success: {success_count}/{len(h5_files)}")
        print(f"  Failed: {len(failed_files)}/{len(h5_files)}")

        if failed_files:
            print(f"\n⚠ Failed files preview:")
            for file, error in failed_files[:5]:
                print(f"  - {file}: {error}")
            if len(failed_files) > 5:
                print(f"  ...and {len(failed_files)-5} more failed files not shown")

        # Create stacked plot if requested
        if create_stacked_plot and success_count > 0:
            print(f"\nGenerating stacked plot...")
            self.create_stacked_plot(output_dir, offset=stacked_plot_offset)

    def _extract_pressure(self, filename):
        """
        Extract pressure value from filename

        Supports both loading and unloading data:
        - Loading: 10, 10.5, 40, etc.
        - Unloading: d3.21, d38.2, etc. (prefix 'd' indicates unloading)

        Returns:
            tuple: (pressure, is_unload) where is_unload is True if filename starts with 'd'
        """
        basename = os.path.basename(filename)
        # Remove file extension
        name_without_ext = os.path.splitext(basename)[0]

        is_unload = False

        # Check if filename starts with 'd' (unloading data)
        if name_without_ext.startswith('d') or name_without_ext.startswith('D'):
            is_unload = True
            # Remove the 'd' prefix for pressure extraction
            name_without_ext = name_without_ext[1:]

        # Try various patterns
        patterns = [
            r'(\d+\.?\d*)[_\s]?GPa',  # 10GPa, 10.5GPa, 10_GPa
            r'[Pp](\d+\.?\d*)',        # P10, p10.5
            r'pressure[_\s]?(\d+\.?\d*)',  # pressure_10
            r'^(\d+\.?\d*)',           # Just numbers at start
        ]

        for pattern in patterns:
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                return float(match.group(1)), is_unload

        return 0.0, is_unload

    def _extract_range_average(self, filename):
        """
        Extract range values from filename and calculate average
        Example: 0.72_Bin001_0.0-10.0.xy -> 5.0
                 0.72_Bin002_10.0-20.0.xy -> 15.0

        Returns:
            float or None: Average of the range, or None if no range found
        """
        basename = os.path.basename(filename)

        # Match pattern like "number-number"
        pattern = r'(\d+\.?\d*)[_\s]*-[_\s]*(\d+\.?\d*)'
        match = re.search(pattern, basename)

        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            average = (start + end) / 2.0
            return average

        return None

    def create_stacked_plot(self, output_dir, offset='auto', output_name='stacked_plot.png'):
        """
        Create stacked diffraction pattern plot
        Supports two modes:
        1. If same pressure has more than 2 data files (e.g., 0.72_Bin001_0.0-10.0.xy), create separate stacked plot for each pressure
        2. Otherwise, use original logic to create one stacked plot for all pressures

        Args:
            output_dir (str): Directory containing .xy or .dat files
            offset (str or float): Offset between curves ('auto' or specific value)
            output_name (str): Output filename
        """
        # Find all .xy or .dat files
        xy_files = glob.glob(os.path.join(output_dir, '*.xy'))
        if not xy_files:
            xy_files = glob.glob(os.path.join(output_dir, '*.dat'))

        if not xy_files:
            print("⚠ No .xy or .dat files found for stacked plot")
            return

        # Group files by pressure
        from collections import defaultdict
        pressure_groups = defaultdict(list)

        for f in xy_files:
            pressure, is_unload = self._extract_pressure(f)
            range_avg = self._extract_range_average(f)
            pressure_groups[pressure].append((f, range_avg))

        # Check if any pressure group contains more than 2 files
        has_multi_file_groups = any(len(files) > 2 for files in pressure_groups.values())

        if has_multi_file_groups:
            # Mode 1: Generate separate stacked plot for each pressure group with multiple files
            print(f"Detected multiple data files at same pressure, generating separate stacked plot for each pressure point")

            for pressure, file_list in sorted(pressure_groups.items()):
                if len(file_list) <= 2:
                    print(f"  Pressure {pressure:.2f} GPa has only {len(file_list)} files, skipping")
                    continue

                print(f"  Generating stacked plot for pressure {pressure:.2f} GPa ({len(file_list)} files)...")
                self._create_single_pressure_stacked_plot(
                    pressure, file_list, output_dir, offset,
                    f'stacked_plot_{pressure:.2f}GPa.png'
                )
        else:
            # Mode 2: Original logic - all pressures in one plot
            print(f"Using original logic to generate stacked plot for all pressures")
            self._create_all_pressure_stacked_plot(xy_files, output_dir, offset, output_name)

    def _create_single_pressure_stacked_plot(self, pressure, file_list, output_dir, offset, output_name):
        """
        Create stacked plot for multiple data files at a single pressure point

        Args:
            pressure (float): Pressure value
            file_list (list): [(file_path, range_average), ...] List of files and their range averages
            output_dir (str): Output directory
            offset (str or float): Offset value
            output_name (str): Output filename
        """
        # Sort by range average
        file_list_sorted = sorted(file_list, key=lambda x: x[1] if x[1] is not None else 0)

        # Load data
        data_list = []
        range_avgs = []
        for file_path, range_avg in file_list_sorted:
            try:
                data = np.loadtxt(file_path)
                data_list.append(data)
                range_avgs.append(range_avg if range_avg is not None else 0)
            except Exception as e:
                print(f"    Warning: Could not load {file_path}: {e}")

        if not data_list:
            print(f"    ⚠ No valid data files for pressure {pressure:.2f} GPa")
            return

        # Calculate offset
        if offset == 'auto':
            max_intensities = [np.max(data[:, 1]) for data in data_list]
            calc_offset = np.mean(max_intensities) * 1.2
        else:
            calc_offset = float(offset)

        # Create figure
        plt.figure(figsize=(12, 10))

        # Use color map that cycles every 90 (e.g., 0-90, 90-180,...)
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))

        for idx, (data, range_avg) in enumerate(zip(data_list, range_avgs)):
            # Plot curve with offset
            y_offset = idx * calc_offset
            color_idx = int(range_avg // 90) if range_avg is not None else idx
            color = base_colors[color_idx % len(base_colors)]
            label = f'{range_avg:.1f}' if range_avg is not None else f'Data {idx+1}'
            plt.plot(data[:, 0], data[:, 1] + y_offset,
                    color=color, linewidth=1.2, label=label)

            # Add label anchored to the offset-adjusted curve
            x_pos = data[0, 0] + (data[-1, 0] - data[0, 0]) * 0.02
            y_pos = y_offset + np.max(data[:, 1]) * 0.3

            plt.text(x_pos, y_pos, label,
                    fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

        plt.xlabel('2θ (degrees)', fontsize=12)
        plt.ylabel('Intensity (offset)', fontsize=12)
        plt.title(f'Stacked Diffraction Patterns at {pressure:.2f} GPa',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Save figure
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Stacked plot saved: {output_path}")
        print(f"      Number of data files: {len(data_list)}")
        print(f"      Offset: {calc_offset:.2f}")

    def _create_all_pressure_stacked_plot(self, xy_files, output_dir, offset, output_name):
        """
        Create overall stacked plot for all pressures

        Supports special ordering for unloading data (files with 'd' prefix):
        - Loading data (no 'd'): sorted by pressure from low to high
        - Unloading data (with 'd'): sorted by pressure from high to low, stacked on top of loading data

        Example: If loading data has max pressure 40 GPa, and unloading data includes d38.2, d30, d20,
                 the stacking order will be: ...38, 39, 40, d38.2, d30, d20

        Args:
            xy_files (list): List of all .xy or .dat files
            output_dir (str): Output directory
            offset (str or float): Offset value
            output_name (str): Output filename
        """
        # Extract pressure, is_unload and sort
        file_info_list = []
        for f in xy_files:
            pressure, is_unload = self._extract_pressure(f)
            file_info_list.append((f, pressure, is_unload))

        # Separate loading and unloading data
        loading_data = [(f, p) for f, p, u in file_info_list if not u]
        unloading_data = [(f, p) for f, p, u in file_info_list if u]

        # Sort loading data: low to high pressure
        loading_data.sort(key=lambda x: x[1])

        # Sort unloading data: high to low pressure
        unloading_data.sort(key=lambda x: x[1], reverse=True)

        # Combine: loading first, then unloading
        file_pressure_pairs = loading_data + unloading_data

        # Load data
        data_list = []
        pressures = []
        is_unload_list = []
        for file_path, pressure in file_pressure_pairs:
            try:
                data = np.loadtxt(file_path)
                data_list.append(data)
                pressures.append(pressure)
                # Determine if this is unload data
                _, is_unload = self._extract_pressure(file_path)
                is_unload_list.append(is_unload)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        if not data_list:
            print("⚠ No valid data files for stacked plot")
            return

        # Calculate offset
        if offset == 'auto':
            # Auto-calculate based on max intensity
            max_intensities = [np.max(data[:, 1]) for data in data_list]
            calc_offset = np.mean(max_intensities) * 1.2
        else:
            calc_offset = float(offset)

        # Create color map (change color every 10 GPa)
        colors = plt.cm.tab10(np.arange(10))

        # Create plot
        plt.figure(figsize=(12, 10))

        for idx, (data, pressure, is_unload) in enumerate(zip(data_list, pressures, is_unload_list)):
            # Determine color based on pressure range
            color_idx = int(pressure // 10) % 10

            # Plot with offset
            y_offset = idx * calc_offset

            # Add 'd' prefix to label if it's unloading data
            label = f'd{pressure:.1f} GPa' if is_unload else f'{pressure:.1f} GPa'

            plt.plot(data[:, 0], data[:, 1] + y_offset,
                    color=colors[color_idx], linewidth=1.2, label=label)

            # Add pressure label on the left side
            # Position it above the baseline of current curve
            x_pos = data[0, 0] + (data[-1, 0] - data[0, 0]) * 0.02
            y_pos = y_offset + np.max(data[:, 1]) * 0.3

            plt.text(x_pos, y_pos, label,
                    fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[color_idx], alpha=0.3))

        plt.xlabel('2θ (degrees)', fontsize=12)
        plt.ylabel('Intensity (offset)', fontsize=12)
        plt.title('Stacked Diffraction Patterns (Loading + Unloading)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Save plot
        output_path = os.path.join(output_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Stacked plot saved: {output_path}")
        print(f"  Total patterns: {len(data_list)}")
        print(f"  Loading data: {len(loading_data)}, Unloading data: {len(unloading_data)}")
        print(f"  Pressure range: {min(pressures):.1f} - {max(pressures):.1f} GPa")
        print(f"  Offset: {calc_offset:.2f}")


def load_config(config_file):
    """Load config file"""
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    
    paths = {
        'poni_file': config.get('paths', 'poni_file'),
        'mask_file': config.get('paths', 'mask_file', fallback=None),
        'input_pattern': config.get('paths', 'input_pattern'),
        'output_dir': config.get('paths', 'output_dir'),
        'dataset_path': config.get('paths', 'dataset_path', fallback=None)
    }
    
    if paths['mask_file'] == '':
        paths['mask_file'] = None
    if paths['dataset_path'] == '':
        paths['dataset_path'] = None
    
    integration = {
        'npt': config.getint('integration', 'npt', fallback=2000),
        'unit': config.get('integration', 'unit', fallback='2th_deg'),
        'correctSolidAngle': config.getboolean('integration', 'correct_solid_angle', fallback=True),
        'polarization_factor': config.get('integration', 'polarization_factor', fallback='None')
    }
    
    if integration['polarization_factor'] == 'None':
        integration['polarization_factor'] = None
    else:
        integration['polarization_factor'] = float(integration['polarization_factor'])
    
    advanced = {
        'method': config.get('advanced', 'method', fallback='csr'),
        'safe': config.getboolean('advanced', 'safe', fallback=True),
        'normalization_factor': config.getfloat('advanced', 'normalization_factor', fallback=1.0)
    }
    
    return paths, integration, advanced

def run_batch_integration(
    poni_file,
    mask_file,
    input_pattern,
    output_dir,
    dataset_path=None,
    npt=2000,
    unit='2th_deg',
    formats=['xy', 'dat', 'chi', 'svg', 'png', 'fxye'],
    create_stacked_plot=True,
    stacked_plot_offset='auto'
):
    """
    Run batch 1D integration using pyFAI

    Args:
        poni_file (str): Path to .poni calibration file
        mask_file (str): Path to mask file (can be None)
        input_pattern (str): Glob pattern to input HDF5 files
        output_dir (str): Output directory
        dataset_path (str, optional): HDF5 dataset path (autodetect if None)
        npt (int): Number of integration points
        unit (str): Output unit (e.g. '2th_deg', 'q_A^-1', etc.)
        formats (list): Output formats ['xy', 'dat', 'chi', 'svg', 'png', 'fxye']
        create_stacked_plot (bool): Whether to create stacked plot
        stacked_plot_offset (str or float): Offset for stacked plot
    """

    integration_kwargs = {
        'correctSolidAngle': True,
        'polarization_factor': None,
        'method': 'csr',
        'safe': True,
        'normalization_factor': 1.0
    }

    if not os.path.exists(poni_file):
        raise FileNotFoundError(f"Calibration file not found: {poni_file}")

    integrator = BatchIntegrator(poni_file, mask_file)

    integrator.batch_integrate(
        input_pattern=input_pattern,
        output_dir=output_dir,
        npt=npt,
        unit=unit,
        dataset_path=dataset_path,
        formats=formats,
        create_stacked_plot=create_stacked_plot,
        stacked_plot_offset=stacked_plot_offset,
        **integration_kwargs
    )
def main():
    """Main function: hardcoded version"""
    print("=" * 80)
    print("HDF5 Diffraction Image Batch Integration Script - Enhanced Version")
    print("Multiple Output Formats + Stacked Pressure Plot")
    print("=" * 80)


    # : 'xy', 'dat', 'chi', 'fxye', 'svg', 'png'

    # 
    output_formats = ['xy']

    run_batch_integration(
        poni_file=r'D:\HEPS\ID31\test\using.poni',
        mask_file=r'D:\HEPS\ID31\test\use.edf',
        input_pattern=r'D:\HEPS\ID31\test\input_dir\*.h5',
        output_dir=r'D:\HEPS\ID31\test\out_dir',
        dataset_path=None,
        npt=2000,
        unit='2th_deg',
        formats=output_formats, create_stacked_plot=True,  #
        stacked_plot_offset='auto'  #
    )


if __name__ == "__main__":
    main()