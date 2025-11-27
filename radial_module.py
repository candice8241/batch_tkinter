#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XRD Azimuthal Integration Module with GUI - Enhanced with Bin Mode
====================================================================
Author: candicewang928@gmail.com
Created: Nov 15, 2025
"""

# Fix Tcl_AsyncDelete threading error: Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter thread conflicts

import os
import glob
import threading
import weakref
from pathlib import Path
from typing import List, Optional, Tuple
import hdf5plugin
import h5py
import numpy as np
import pandas as pd
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog

from theme_module import GUIBase, CuteSheepProgressBar, ModernTab, ModernButton
from half_auto_fitting import PeakFittingGUI
from batch_integration import BatchIntegrator


# ============================================================================
# Custom UI Components
# ============================================================================

class SpinboxStyleButton(tk.Frame):
    """Spinbox-style button widget matching the reference image"""

    def __init__(self, parent, text, command, width=80, font_size=9, **kwargs):
        super().__init__(parent, bg='#E8D5F0', **kwargs)

        self.command = command
        self.text = text

        # Configure frame with rounded appearance
        self.configure(relief='solid', borderwidth=1, highlightbackground='#C8B5D0',
                      highlightthickness=1)

        # Create button
        self.button = tk.Button(
            self,
            text=text,
            command=command,
            bg='#E8D5F0',
            fg='#6B4C7A',
            font=('Arial', font_size),
            relief='flat',
            borderwidth=0,
            activebackground='#D5C0E0',
            activeforeground='#6B4C7A',
            cursor='hand2',
            padx=15,
            pady=5
        )
        self.button.pack(fill=tk.BOTH, expand=True)

        # Hover effects
        self.button.bind('<Enter>', self._on_enter)
        self.button.bind('<Leave>', self._on_leave)

    def _on_enter(self, event):
        self.button.config(bg='#D5C0E0')
        self.configure(bg='#D5C0E0')

    def _on_leave(self, event):
        self.button.config(bg='#E8D5F0')
        self.configure(bg='#E8D5F0')


class CustomSpinbox(tk.Frame):
    """Custom spinbox with left/right arrow buttons for value adjustment"""

    def __init__(self, parent, from_=0, to=100, textvariable=None, increment=1,
                 width=80, is_float=False, **kwargs):
        super().__init__(parent, bg='#F0E6FA', **kwargs)

        self.from_ = from_
        self.to = to
        self.increment = increment
        self.is_float = is_float
        self.textvariable = textvariable

        # Left decrease button
        self.left_btn = tk.Button(
            self,
            text="<",
            command=self.decrease,
            bg='white',
            fg='#6B4C7A',
            font=('Arial', 10, 'bold'),
            relief='solid',
            borderwidth=1,
            activebackground='#F0E6FA',
            cursor='hand2',
            width=2,
            padx=2,
            pady=2
        )
        self.left_btn.pack(side=tk.LEFT, padx=2)

        # Value display
        self.value_frame = tk.Frame(self, bg='white', relief='solid', borderwidth=1)
        self.value_frame.pack(side=tk.LEFT, padx=2)

        self.entry = tk.Entry(
            self.value_frame,
            textvariable=textvariable,
            font=('Arial', 10),
            bg='white',
            fg='#333333',
            justify='center',
            relief='flat',
            borderwidth=0,
            width=8
        )
        self.entry.pack(padx=3, pady=2)

        # Right increase button
        self.right_btn = tk.Button(
            self,
            text=">",
            command=self.increase,
            bg='white',
            fg='#6B4C7A',
            font=('Arial', 10, 'bold'),
            relief='solid',
            borderwidth=1,
            activebackground='#F0E6FA',
            cursor='hand2',
            width=2,
            padx=2,
            pady=2
        )
        self.right_btn.pack(side=tk.LEFT, padx=2)

    def increase(self):
        """Increase value"""
        try:
            current = self.textvariable.get()
            if self.is_float:
                new_val = min(float(self.to), current + self.increment)
            else:
                new_val = min(int(self.to), current + self.increment)
            self.textvariable.set(new_val)
        except:
            pass

    def decrease(self):
        """Decrease value"""
        try:
            current = self.textvariable.get()
            if self.is_float:
                new_val = max(float(self.from_), current - self.increment)
            else:
                new_val = max(int(self.from_), current - self.increment)
            self.textvariable.set(new_val)
        except:
            pass


# ============================================================================
# Core Integration Class (unchanged)
# ============================================================================

class XRDAzimuthalIntegrator:
    """Class to handle azimuthal integration of XRD diffraction data."""

    def __init__(self, poni_file: str, mask_file: Optional[str] = None):
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.ai = None
        self.mask = None
        self._load_calibration()
        if mask_file:
            self._load_mask()

    def _load_calibration(self):
        if not os.path.exists(self.poni_file):
            raise FileNotFoundError(f"PONI file not found: {self.poni_file}")
        print(f"Loading calibration from: {self.poni_file}")
        self.ai = pyFAI.load(self.poni_file)
        print(f"  Detector: {self.ai.detector.name}")
        print(f"  Distance: {self.ai.dist * 1000:.2f} mm")
        print(f"  Wavelength: {self.ai.wavelength * 1e10:.4f} Å")

    def _load_mask(self):
        if not os.path.exists(self.mask_file):
            print(f"Warning: Mask file not found: {self.mask_file}")
            return
        print(f"Loading mask from: {self.mask_file}")
        ext = os.path.splitext(self.mask_file)[1].lower()
        if ext == '.npy':
            self.mask = np.load(self.mask_file)
        elif ext in ['.edf', '.tif', '.tiff']:
            try:
                import fabio
                img = fabio.open(self.mask_file)
                self.mask = img.data
            except ImportError:
                print("Warning: fabio not installed. Cannot read mask file.")
                return
        elif ext in ['.h5', '.hdf5']:
            with h5py.File(self.mask_file, 'r') as f:
                for key in ['mask', 'data', 'entry/data/data']:
                    if key in f:
                        self.mask = f[key][:]
                        break
                else:
                    keys = list(f.keys())
                    if keys:
                        self.mask = f[keys[0]][:]
        else:
            print(f"Warning: Unsupported mask file format: {ext}")
            return
        print(f"  Mask shape: {self.mask.shape}")
        print(f"  Masked pixels: {np.sum(self.mask)}")

    def integrate_file(self, h5_file: str, output_dir: str,
                      npt: int = 2048, unit: str = "q_A^-1",
                      output_format: str = "xy",
                      azimuth_range: Optional[tuple] = None,
                      sector_label: str = "",
                      dataset_path: Optional[str] = None) -> Tuple[str, np.ndarray, np.ndarray]:
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
        print(f"\nProcessing: {os.path.basename(h5_file)}")
        data = self._read_h5_data(h5_file, dataset_path=dataset_path)
        print(f"  Integrating with {npt} points, unit={unit}")
        if azimuth_range:
            print(f"  Azimuthal range: {azimuth_range[0]}° to {azimuth_range[1]}°")
        result = self.ai.integrate1d(data, npt=npt, mask=self.mask, unit=unit,
                                     method="splitpixel", error_model="poisson",
                                     azimuth_range=azimuth_range)
        if isinstance(result, tuple):
            if len(result) == 2:
                q, intensity = result
            elif len(result) == 3:
                q, intensity, sigma = result
            else:
                q = result[0]
                intensity = result[1]
        else:
            q = result.radial
            intensity = result.intensity
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        if sector_label:
            output_file = os.path.join(output_dir, f"{base_name}_{sector_label}.{output_format}")
        else:
            output_file = os.path.join(output_dir, f"{base_name}_integrated.{output_format}")
        self._save_data(q, intensity, output_file, unit, output_format)
        print(f"  Saved: {output_file}")
        return output_file, q, intensity

    def _read_h5_data(self, h5_file: str, dataset_path: str = None) -> np.ndarray:
        with h5py.File(h5_file, 'r') as f:
            if dataset_path and dataset_path in f:
                data = f[dataset_path][...]
                if data.ndim == 3:
                    data = data[0]
                return data
            common_paths = ['entry/data/data', 'entry/instrument/detector/data',
                          'data', 'image', 'diffraction']
            data = None
            for path in common_paths:
                if path in f:
                    data = f[path][...]
                    if data.ndim == 3:
                        data = data[0]
                    break
            if data is None:
                def find_2d_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if obj.ndim == 2 or (obj.ndim == 3 and obj.shape[0] == 1):
                            return name
                    return None
                for key in f.keys():
                    result = f[key].visititems(find_2d_dataset)
                    if result:
                        data = f[result][...]
                        if data.ndim == 3:
                            data = data[0]
                        break
                if data is None:
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][...]
                        if data.ndim == 3:
                            data = data[0]
            if data is None:
                raise ValueError(f"No suitable dataset found in {h5_file}")
            print(f"  Data shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Intensity range: [{np.min(data):.1f}, {np.max(data):.1f}]")
            return data

    def _save_data(self, q: np.ndarray, intensity: np.ndarray,
                   output_file: str, unit: str, output_format: str):
        if output_format == "xy":
            header = f"# Azimuthal integration\n# Unit: {unit}\n# Column 1: {unit}\n# Column 2: Intensity"
            np.savetxt(output_file, np.column_stack([q, intensity]),
                      header=header, fmt='%.6e')
        elif output_format == "chi":
            with open(output_file, 'w') as f:
                f.write(f"2-Theta Angle (Degrees)\n")
                f.write(f"Intensity\n")
                for q_val, int_val in zip(q, intensity):
                    f.write(f"{q_val:.6f} {int_val:.6f}\n")
        elif output_format == "dat":
            errors = np.sqrt(np.maximum(intensity, 1))
            header = f"# Azimuthal integration\n# Unit: {unit}\n# Column 1: {unit}\n# Column 2: Intensity\n# Column 3: Error"
            np.savetxt(output_file, np.column_stack([q, intensity, errors]),
                      header=header, fmt='%.6e')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def batch_process(self, h5_files: List[str], output_dir: str, **kwargs) -> List[str]:
        output_files = []
        total = len(h5_files)
        print(f"\n{'='*60}")
        print(f"Batch processing {total} file(s)")
        print(f"{'='*60}")
        for i, h5_file in enumerate(h5_files, 1):
            print(f"\n[{i}/{total}]", end=" ")
            try:
                output_file, _, _ = self.integrate_file(h5_file, output_dir, **kwargs)
                output_files.append(output_file)
            except Exception as e:
                print(f"  Error processing {h5_file}: {e}")
                continue
        print(f"\n{'='*60}")
        print(f"Completed: {len(output_files)}/{total} files processed successfully")
        print(f"{'='*60}\n")
        return output_files


# ============================================================================
# GUI Module with Thread Safety (Fixed for Jupyter/IPython)
# ============================================================================

class AzimuthalIntegrationModule(GUIBase):
    """Azimuthal Integration module - Thread-safe version"""

    def __init__(self, parent, root):
        super().__init__()
        self.parent = parent
        self.root = root
        self.main_frame = None
        self._content_parent = None
        self._cleanup_lock = threading.Lock()
        self._is_destroyed = False
        self._init_variables()
        self.processing = False
        self.stop_processing = False
        self.custom_sectors = []
        self.sector_row_widgets = []

        # Track trace IDs for proper cleanup
        self._trace_ids = []

        # Register cleanup on window close
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        except:
            pass

    def _on_closing(self):
        """Clean shutdown handler"""
        with self._cleanup_lock:
            self._is_destroyed = True
            self.stop_processing = True

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def _init_variables(self):
        """Initialize all tkinter variables with thread safety"""
        try:
            self.poni_path = tk.StringVar()
            self.mask_path = tk.StringVar()
            self.input_pattern = tk.StringVar()
            self.output_dir = tk.StringVar()
            self.dataset_path = tk.StringVar(value="entry/data/data")
            self.npt = tk.IntVar(value=4000)
            self.unit = tk.StringVar(value='2th_deg')
            self.azimuth_start = tk.DoubleVar(value=0.0)
            self.azimuth_end = tk.DoubleVar(value=90.0)
            self.sector_label = tk.StringVar(value="Sector_1")
            self.preset = tk.StringVar(value='quadrants')
            self.mode = tk.StringVar(value='single')
            self.multiple_mode = tk.StringVar(value='custom')
            self.output_csv = tk.BooleanVar(value=True)

            # Bin mode variables
            self.bin_mode = tk.BooleanVar(value=False)
            self.bin_start = tk.DoubleVar(value=0.0)
            self.bin_end = tk.DoubleVar(value=360.0)
            self.bin_step = tk.DoubleVar(value=10.0)

            # Multi bin mode
            self.multi_bin_mode = tk.BooleanVar(value=False)

            # Output format options (6 formats)
            self.format_xy = tk.BooleanVar(value=True)
            self.format_dat = tk.BooleanVar(value=False)
            self.format_chi = tk.BooleanVar(value=False)
            self.format_fxye = tk.BooleanVar(value=False)
            self.format_svg = tk.BooleanVar(value=False)
            self.format_png = tk.BooleanVar(value=False)

            # Stacked plot options
            self.create_stacked_plot = tk.BooleanVar(value=False)
            self.stacked_plot_offset = tk.StringVar(value='auto')
        except Exception as e:
            print(f"Warning: Error initializing variables: {e}")

    def _remove_all_traces(self):
        """Remove all registered trace callbacks to prevent accumulation"""
        for var, trace_id in self._trace_ids:
            try:
                var.trace_remove('write', trace_id)
            except:
                pass
        self._trace_ids = []

    def _add_trace(self, var, callback):
        """Add a trace and register it for later cleanup"""
        try:
            trace_id = var.trace_add('write', callback)
            self._trace_ids.append((var, trace_id))
        except:
            pass

    def setup_ui(self):
        """Setup UI with error handling"""
        if self.main_frame is not None and self.main_frame.winfo_exists():
            self.main_frame.pack(fill=tk.BOTH, expand=True)
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            return

        self.main_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        self._content_parent = self.main_frame

        self._create_reference_section()
        self._create_separated_settings_sections()
        self._create_output_options_section()
        self._create_progress_section()
        self._create_log_section()

        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # Update UI after all widgets are created
        try:
            self.parent.update_idletasks()
        except:
            pass

    def _section_parent(self):
        if self._content_parent is not None and self._content_parent.winfo_exists():
            return self._content_parent
        return self.parent

    def _create_reference_section(self):
        """Reference with larger font and CENTERED"""
        ref_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        ref_frame.pack(fill=tk.X, padx=0, pady=(10, 10))

        ref_card = self.create_card_frame(ref_frame)
        ref_card.pack(fill=tk.X)

        content = tk.Frame(ref_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.X)

        center_container = tk.Frame(content, bg=self.colors['card_bg'])
        center_container.pack(expand=True)

        tk.Label(center_container, text="🍓 Azimuthal Angle Reference:",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 10, 'bold')).pack()

        ref_text = "0° = Right (→)  |  90° = Top (↑)  |  180° = Left (←)  |  270° = Bottom (↓)"
        tk.Label(center_container, text=ref_text,
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Arial', 10)).pack(pady=(5, 0))

        tk.Label(center_container, text="Counter-clockwise rotation from right horizontal",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Arial', 9, 'italic')).pack()

    def _create_separated_settings_sections(self):
        """Create two independent modules side by side: Integration Settings (left) and Azimuthal Angle Settings (right)"""
        # Main container for both sections
        sections_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        sections_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        # Container for left-right layout
        layout_container = tk.Frame(sections_frame, bg=self.colors['bg'])
        layout_container.pack(fill=tk.BOTH, expand=True)

        # ========== LEFT MODULE: Integration Settings ==========
        left_module = tk.Frame(layout_container, bg=self.colors['bg'], height=450)
        left_module.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_module.pack_propagate(False)

        left_card = self.create_card_frame(left_module)
        left_card.pack(fill=tk.BOTH, expand=True)

        left_content = tk.Frame(left_card, bg=self.colors['card_bg'], padx=20, pady=12)
        left_content.pack(fill=tk.BOTH, expand=True)

        # Left header
        left_header = tk.Frame(left_content, bg=self.colors['card_bg'])
        left_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(left_header, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(left_header, text="Integration Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # File pickers
        self.create_file_picker_with_spinbox_btn(left_content, "PONI File", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(left_content, "Mask File", self.mask_path,
                               [("Mask files", "*.npy *.h5 *.edf"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(left_content, "Input .h5 File",
                               self.input_pattern, [("HDF5 files", "*.h5"), ("All files", "*.*")])
        self.create_folder_picker_with_spinbox_btn(left_content, "Output Directory", self.output_dir)

        # Dataset Path
        dataset_container = tk.Frame(left_content, bg=self.colors['card_bg'])
        dataset_container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(dataset_container, text="Dataset Path", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        dataset_input_frame = tk.Frame(dataset_container, bg=self.colors['card_bg'])
        dataset_input_frame.pack(fill=tk.X)

        tk.Entry(dataset_input_frame, textvariable=self.dataset_path, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        dataset_browse_btn = SpinboxStyleButton(
            dataset_input_frame,
            "Browse",
            lambda: self.browse_dataset_path(),
            width=75
        )
        dataset_browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Parameters - Number of Points and Unit in same row
        param_frame = tk.Frame(left_content, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(10, 0))

        # Number of Points (left side with expand for distributed centering)
        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, expand=True)
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        CustomSpinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                     increment=100, is_float=False).pack(anchor=tk.W)

        # Unit (right side with expand for distributed centering)
        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, expand=True)

        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(pady=(0, 8))

        unit_options_frame = tk.Frame(unit_cont, bg=self.colors['card_bg'])
        unit_options_frame.pack()

        tk.Radiobutton(unit_options_frame, text="2θ (°)", variable=self.unit, value='2th_deg',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="Q (Å⁻¹)", variable=self.unit, value='q_A^-1',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="r (mm)", variable=self.unit, value='r_mm',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT)

        # ========== RIGHT MODULE: Azimuthal Angle Settings ==========
        right_module = tk.Frame(layout_container, bg=self.colors['bg'], height=450)
        right_module.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_module.pack_propagate(False)

        right_card = self.create_card_frame(right_module)
        right_card.pack(fill=tk.BOTH, expand=True)

        right_content = tk.Frame(right_card, bg=self.colors['card_bg'], padx=20, pady=12)
        right_content.pack(fill=tk.BOTH, expand=True)

        # Right header
        right_header = tk.Frame(right_content, bg=self.colors['card_bg'])
        right_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(right_header, text="🍰", bg=self.colors['card_bg'],
                font=('Arial', 16)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(right_header, text="Azimuthal Angle Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Warning box (Figure 1)
        warning_box = tk.Frame(right_content, bg='#FFF4DC', relief='solid', borderwidth=1, padx=8, pady=4)
        warning_box.pack(pady=(10, 0))

        tk.Label(warning_box, text="💡 Define sectors with bin size. Each sector will be divided into multiple bins.",
                bg='#FFF4DC', fg=self.colors['text_dark'],
                font=('Arial', 8)).pack()

        # Integration Mode (Figure 3)
        mode_container = tk.Frame(right_content, bg=self.colors['card_bg'])
        mode_container.pack(fill=tk.X, pady=(12, 0))

        tk.Label(mode_container, text="Integration Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        mode_buttons = tk.Frame(mode_container, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Single Sector", variable=self.mode,
                      value='single', bg=self.colors['card_bg'],
                      font=('Arial', 9, 'bold'),
                      command=self.update_mode).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(mode_buttons, text="Multiple Sectors", variable=self.mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Arial', 9, 'bold'),
                      command=self.update_mode).pack(side=tk.LEFT)

        # Dynamic frame for mode-specific content
        self.dynamic_frame = tk.Frame(right_content, bg=self.colors['card_bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.update_mode()

    def _create_merged_settings_section(self):
        """Merged card with Integration Settings (left) and Azimuthal Angle Settings (right)"""
        merged_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        merged_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        # Card frame
        card = self.create_card_frame(merged_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Integration Settings & Azimuthal Angle Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Container for left-right layout
        main_container = tk.Frame(content, bg=self.colors['card_bg'])
        main_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # ========== LEFT SECTION: Integration Settings ==========
        left_section = tk.Frame(main_container, bg=self.colors['card_bg'])
        left_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 30))

        tk.Label(left_section, text="Integration Settings", bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        # File pickers
        self.create_file_picker_with_spinbox_btn(left_section, "PONI File", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(left_section, "Mask File", self.mask_path,
                               [("Mask files", "*.npy *.h5 *.edf"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(left_section, "Input .h5 File",
                               self.input_pattern, [("HDF5 files", "*.h5"), ("All files", "*.*")])
        self.create_folder_picker_with_spinbox_btn(left_section, "Output Directory", self.output_dir)

        # Dataset Path
        dataset_container = tk.Frame(left_section, bg=self.colors['card_bg'])
        dataset_container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(dataset_container, text="Dataset Path", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        dataset_input_frame = tk.Frame(dataset_container, bg=self.colors['card_bg'])
        dataset_input_frame.pack(fill=tk.X)

        tk.Entry(dataset_input_frame, textvariable=self.dataset_path, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        dataset_browse_btn = SpinboxStyleButton(
            dataset_input_frame,
            "Browse",
            lambda: self.browse_dataset_path(),
            width=75
        )
        dataset_browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Parameters - Number of Points and Unit in same row
        param_frame = tk.Frame(left_section, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(10, 0))

        # Number of Points (left side with expand for distributed centering)
        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, expand=True)
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        CustomSpinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                     increment=100, is_float=False).pack(anchor=tk.W)

        # Unit (right side with expand for distributed centering)
        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, expand=True)

        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(pady=(0, 8))

        unit_options_frame = tk.Frame(unit_cont, bg=self.colors['card_bg'])
        unit_options_frame.pack()

        tk.Radiobutton(unit_options_frame, text="2θ (°)", variable=self.unit, value='2th_deg',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="Q (Å⁻¹)", variable=self.unit, value='q_A^-1',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="r (mm)", variable=self.unit, value='r_mm',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT)

        # ========== RIGHT SECTION: Azimuthal Angle Settings ==========
        right_outer = tk.Frame(main_container, bg=self.colors['card_bg'], width=450)
        right_outer.pack(side=tk.LEFT, fill=tk.Y)
        right_outer.pack_propagate(False)

        # Create vertical centering container
        center_container = tk.Frame(right_outer, bg=self.colors['card_bg'])
        center_container.pack(fill=tk.BOTH, expand=True)

        # Top padding
        tk.Frame(center_container, bg=self.colors['card_bg']).pack(expand=True)

        # Content area (horizontally and vertically centered)
        right_section = tk.Frame(center_container, bg=self.colors['card_bg'])
        right_section.pack()

        # Azimuthal settings outer border frame
        azimuthal_border = tk.Frame(right_section, bg=self.colors['card_bg'],
                                     relief='solid', borderwidth=1)
        azimuthal_border.pack(fill=tk.BOTH)

        # Azimuthal settings content area (with padding)
        azimuthal_content = tk.Frame(azimuthal_border, bg=self.colors['card_bg'])
        azimuthal_content.pack(fill=tk.BOTH, padx=10, pady=10)

        # Right side title
        tk.Label(azimuthal_content, text="Azimuthal Angle Settings", bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        # Mode selection with warning box
        mode_container = tk.Frame(azimuthal_content, bg=self.colors['card_bg'])
        mode_container.pack(fill=tk.X, pady=(0, 10))

        tk.Label(mode_container, text="Integration Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        mode_buttons = tk.Frame(mode_container, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Single Sector", variable=self.mode,
                      value='single', bg=self.colors['card_bg'],
                      font=('Arial', 9, 'bold'),
                      command=self.update_mode).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(mode_buttons, text="Multiple Sectors", variable=self.mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Arial', 9, 'bold'),
                      command=self.update_mode).pack(side=tk.LEFT)

        # Warning box
        warning_box = tk.Frame(mode_container, bg='#FFF4DC', relief='solid', borderwidth=1, padx=8, pady=4)
        warning_box.pack(fill=tk.X, pady=(8, 0))

        tk.Label(warning_box, text="💡 Define sectors with bin size. Each sector will be divided into multiple bins.",
                bg='#FFF4DC', fg=self.colors['text_dark'],
                font=('Arial', 8)).pack()

        # Dynamic frame for mode-specific content
        self.dynamic_frame = tk.Frame(azimuthal_content, bg=self.colors['card_bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.update_mode()

        # Bottom padding
        tk.Frame(center_container, bg=self.colors['card_bg']).pack(expand=True)

    def _create_io_section(self):
        """File Configuration section - matching powder_module style"""
        io_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        io_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        # Card frame
        card = self.create_card_frame(io_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Integration Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # File pickers
        self.create_file_picker_with_spinbox_btn(content, "PONI File", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(content, "Mask File", self.mask_path,
                               [("Mask files", "*.npy *.h5 *.edf"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(content, "Input .h5 File",
                               self.input_pattern, [("HDF5 files", "*.h5"), ("All files", "*.*")])
        self.create_folder_picker_with_spinbox_btn(content, "Output Directory", self.output_dir)

        # Dataset Path
        dataset_container = tk.Frame(content, bg=self.colors['card_bg'])
        dataset_container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(dataset_container, text="Dataset Path", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        dataset_input_frame = tk.Frame(dataset_container, bg=self.colors['card_bg'])
        dataset_input_frame.pack(fill=tk.X)

        tk.Entry(dataset_input_frame, textvariable=self.dataset_path, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        dataset_browse_btn = SpinboxStyleButton(
            dataset_input_frame,
            "Browse",
            lambda: self.browse_dataset_path(),
            width=75
        )
        dataset_browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Parameters - Number of Points and Unit in same row
        param_frame = tk.Frame(content, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(10, 0))

        # Number of Points (left side with expand for distributed centering)
        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, expand=True)
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        CustomSpinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                     increment=100, is_float=False).pack(anchor=tk.W)

        # Unit (right side with expand for distributed centering)
        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, expand=True)

        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(pady=(0, 8))

        unit_options_frame = tk.Frame(unit_cont, bg=self.colors['card_bg'])
        unit_options_frame.pack()

        tk.Radiobutton(unit_options_frame, text="2θ (°)", variable=self.unit, value='2th_deg',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="Q (Å⁻¹)", variable=self.unit, value='q_A^-1',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="r (mm)", variable=self.unit, value='r_mm',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT)

    def _create_simple_file_row(self, parent, label_text, variable, browse_command):
        """Create a simple file input row without extra frames"""
        row_frame = tk.Frame(parent, bg='white')
        row_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        # Label
        tk.Label(row_frame, text=label_text,
                bg='white', fg='#555',
                font=('Arial', 10)).pack(anchor=tk.W, pady=(0, 3))

        # Entry and button row
        entry_row = tk.Frame(row_frame, bg='white')
        entry_row.pack(fill=tk.X)

        entry = tk.Entry(entry_row, textvariable=variable,
                        font=('Arial', 10),
                        bg='white', relief='solid', borderwidth=1)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        if browse_command:
            tk.Button(entry_row, text="Browse",
                     command=browse_command,
                     bg='#D8A7D8', fg='white',
                     font=('Arial', 9, 'bold'),
                     relief='flat', padx=12, pady=4,
                     cursor='hand2').pack(side=tk.LEFT, padx=(5, 0))

    def _adjust_npt(self, delta):
        """Adjust number of points"""
        current = self.npt.get()
        new_value = max(100, current + delta)
        self.npt.set(new_value)

    def _browse_file(self, var, description, filetypes):
        """Helper method for file browsing with error handling"""
        try:
            filename = filedialog.askopenfilename(
                title=f"Select {description}",
                filetypes=[(description, filetypes), ("All files", "*.*")]
            )
            if filename:
                var.set(filename)
        except Exception as e:
            self.log(f"Error browsing file: {e}")

    def _select_input_files(self):
        """Helper method for selecting input H5 files with error handling"""
        try:
            files = filedialog.askopenfilenames(
                title="Select Input H5 Files",
                filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
            )
            if files:
                directory = os.path.dirname(files[0])
                pattern = os.path.join(directory, "*.h5")
                self.input_pattern.set(pattern)
        except Exception as e:
            self.log(f"Error selecting files: {e}")

    def _browse_directory(self):
        """Helper method for directory browsing with error handling"""
        try:
            directory = filedialog.askdirectory(title="Select Output Directory")
            if directory:
                self.output_dir.set(directory)
        except Exception as e:
            self.log(f"Error browsing directory: {e}")

    def create_file_picker_with_spinbox_btn(self, parent, label_text, var, filetypes, pattern=False):
        """Create file picker with spinbox-style button"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(container, text=label_text, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        tk.Entry(input_frame, textvariable=var, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        if pattern:
            btn = SpinboxStyleButton(input_frame, "Browse Folder",
                                    lambda: self.browse_folder(var),
                                    width=95)
        else:
            btn = SpinboxStyleButton(input_frame, "Browse",
                                    lambda: self.browse_file(var, filetypes),
                                    width=75)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_folder_picker_with_spinbox_btn(self, parent, label_text, var):
        """Create folder picker with spinbox-style button"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(container, text=label_text, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        tk.Entry(input_frame, textvariable=var, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        btn = SpinboxStyleButton(input_frame, "Browse",
                                lambda: self.browse_folder(var),
                                width=75)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def browse_file(self, var, filetypes):
        """Browse for file"""
        try:
            filename = filedialog.askopenfilename(filetypes=filetypes)
            if filename:
                # If input file (input_pattern), auto-convert to wildcard pattern to traverse the entire folder
                if var == self.input_pattern:
                    directory = os.path.dirname(filename)
                    extension = os.path.splitext(filename)[1]
                    # Create wildcard pattern to match all files with the same extension in the folder
                    pattern = os.path.join(directory, f"*{extension}")
                    var.set(pattern)
                    self.log(f"📂 Input pattern set to: {pattern}")
                    self.log(f"   This will process all {extension} files in the directory")
                else:
                    var.set(filename)
        except Exception as e:
            self.log(f"Error browsing file: {str(e)}")

    def browse_folder(self, var):
        """Browse for folder"""
        try:
            foldername = filedialog.askdirectory()
            if foldername:
                var.set(foldername)
        except Exception as e:
            self.log(f"Error browsing folder: {str(e)}")

    def browse_dataset_path(self):
        """Browse for dataset path - Using simpledialog"""
        result = messagebox.askquestion(
            "Dataset Path",
            "Dataset path is typically an HDF5 internal path like 'entry/data/data'.\n\n" +
            "Do you want to manually enter the path?\n\n" +
            "Click 'No' to keep the current value.",
            icon='question'
        )

        if result == 'yes':
            # Use simpledialog which is thread-safe
            new_path = simpledialog.askstring(
                "Enter Dataset Path",
                "Enter HDF5 Dataset Path:",
                initialvalue=self.dataset_path.get(),
                parent=self.root
            )
            if new_path:
                self.dataset_path.set(new_path)

    def _create_azimuthal_section(self):
        """Azimuthal settings"""
        azimuth_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        azimuth_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        card = self.create_card_frame(azimuth_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 10))

        tk.Label(header, text="🍰", bg=self.colors['card_bg'],
                font=('Arial', 16)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Azimuthal Angle Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        # Mode selection with warning box in same row
        mode_container = tk.Frame(content, bg=self.colors['card_bg'])
        mode_container.pack(fill=tk.X, pady=(0, 15))

        # Left side: Mode selection
        mode_frame = tk.Frame(mode_container, bg=self.colors['card_bg'])
        mode_frame.pack(side=tk.LEFT)

        tk.Label(mode_frame, text="Integration Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 6))

        mode_buttons = tk.Frame(mode_frame, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Single Sector", variable=self.mode,
                      value='single', bg=self.colors['card_bg'],
                      font=('Arial', 10, 'bold'),
                      command=self.update_mode).pack(side=tk.LEFT, padx=(0, 25))

        tk.Radiobutton(mode_buttons, text="Multiple Sectors", variable=self.mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Arial', 10, 'bold'),
                      command=self.update_mode).pack(side=tk.LEFT)

        # Right side: Reference warning box (centered vertically)
        warning_container = tk.Frame(mode_container, bg=self.colors['card_bg'])
        warning_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(30, 0))

        # Center vertically
        center_frame = tk.Frame(warning_container, bg=self.colors['card_bg'])
        center_frame.pack(expand=True)

        warning_box = tk.Frame(center_frame, bg='#FFF4DC', relief='solid', borderwidth=1, padx=10, pady=6)
        warning_box.pack()

        tk.Label(warning_box, text="💡 Define sectors with bin size. Each sector will be divided into multiple bins.",
                bg='#FFF4DC', fg=self.colors['text_dark'],
                font=('Arial', 9, 'bold')).pack()

        self.dynamic_frame = tk.Frame(content, bg=self.colors['card_bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        self.update_mode()

    def _create_output_options_section(self):
        """Output format and stacked plot options section with Run button on the right"""
        # Main container for both sections
        sections_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        sections_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        # Container for left-right layout
        layout_container = tk.Frame(sections_frame, bg=self.colors['bg'])
        layout_container.pack(fill=tk.BOTH, expand=True)

        # ========== LEFT MODULE: Output Options ==========
        left_module = tk.Frame(layout_container, bg=self.colors['bg'], height=200)
        left_module.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_module.pack_propagate(False)

        left_card = self.create_card_frame(left_module)
        left_card.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(left_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="🎨", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Output Options",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Main horizontal container for left-right layout (centered)
        main_container = tk.Frame(content, bg=self.colors['card_bg'])
        main_container.pack(expand=True)

        # ========== LEFT SECTION: Output Formats ==========
        left_section = tk.Frame(main_container, bg=self.colors['card_bg'])
        left_section.pack(side=tk.LEFT, padx=(0, 50))

        # Output Formats title
        tk.Label(left_section, text="Select Output Formats:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        # Output Formats with border (only frames the six format checkboxes)
        formats_border_frame = tk.Frame(left_section, bg=self.colors['card_bg'],
                                       relief='solid', borderwidth=1, highlightbackground='#CCCCCC')
        formats_border_frame.pack()

        # Inner padding frame
        formats_content = tk.Frame(formats_border_frame, bg=self.colors['card_bg'])
        formats_content.pack(padx=8, pady=8)

        # First row of formats (3 checkboxes)
        row1 = tk.Frame(formats_content, bg=self.colors['card_bg'])
        row1.pack(pady=3)

        tk.Checkbutton(row1, text=".xy", variable=self.format_xy, bg=self.colors['card_bg'],
                      font=('Arial', 9), fg=self.colors['text_dark'],
                      selectcolor='#E8D5F0', activebackground=self.colors['card_bg']
                      ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Checkbutton(row1, text=".dat", variable=self.format_dat, bg=self.colors['card_bg'],
                      font=('Arial', 9), fg=self.colors['text_dark'],
                      selectcolor='#E8D5F0', activebackground=self.colors['card_bg']
                      ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Checkbutton(row1, text=".chi", variable=self.format_chi, bg=self.colors['card_bg'],
                      font=('Arial', 9), fg=self.colors['text_dark'],
                      selectcolor='#E8D5F0', activebackground=self.colors['card_bg']
                      ).pack(side=tk.LEFT)

        # Second row of formats (3 checkboxes)
        row2 = tk.Frame(formats_content, bg=self.colors['card_bg'])
        row2.pack(pady=3)

        tk.Checkbutton(row2, text=".fxye", variable=self.format_fxye, bg=self.colors['card_bg'],
                      font=('Arial', 9), fg=self.colors['text_dark'],
                      selectcolor='#E8D5F0', activebackground=self.colors['card_bg']
                      ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Checkbutton(row2, text=".svg", variable=self.format_svg, bg=self.colors['card_bg'],
                      font=('Arial', 9), fg=self.colors['text_dark'],
                      selectcolor='#E8D5F0', activebackground=self.colors['card_bg']
                      ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Checkbutton(row2, text=".png", variable=self.format_png, bg=self.colors['card_bg'],
                      font=('Arial', 9), fg=self.colors['text_dark'],
                      selectcolor='#E8D5F0', activebackground=self.colors['card_bg']
                      ).pack(side=tk.LEFT)

        # ========== RIGHT SECTION: Stacked Plot Options ==========
        right_section = tk.Frame(main_container, bg=self.colors['card_bg'])
        right_section.pack(side=tk.LEFT)

        # Stacked Plot Options title
        tk.Label(right_section, text="Stacked Plot Options:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        # Checkbox on first line
        tk.Checkbutton(right_section, text="Create Stacked Plot",
                      variable=self.create_stacked_plot,
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(anchor=tk.W, pady=(0, 8))

        # Offset section on second line
        offset_container = tk.Frame(right_section, bg=self.colors['card_bg'])
        offset_container.pack(anchor=tk.W)

        tk.Label(offset_container, text="Offset:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))

        offset_entry = tk.Entry(offset_container, textvariable=self.stacked_plot_offset,
                               font=('Arial', 10), width=12, justify='center',
                               bg='white', relief='solid', borderwidth=1)
        offset_entry.pack(side=tk.LEFT, ipady=2)

        # Help text below
        tk.Label(right_section, text="(use 'auto' or number for offset)",
                bg=self.colors['card_bg'], fg='#888888',
                font=('Arial', 8, 'italic')).pack(anchor=tk.W, pady=(2, 0))

        # ========== RIGHT MODULE: Action Buttons Card ==========
        right_module = tk.Frame(layout_container, bg=self.colors['bg'], height=200)
        right_module.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_module.pack_propagate(False)

        # Create card for buttons
        button_card = self.create_card_frame(right_module)
        button_card.pack(fill=tk.BOTH, expand=True)

        # Card content with padding
        card_content = tk.Frame(button_card, bg=self.colors['card_bg'])
        card_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Header with icon
        header_frame = tk.Frame(card_content, bg=self.colors['card_bg'])
        header_frame.pack(pady=(0, 10))

        tk.Label(header_frame, text="⚡", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(header_frame, text="Quick Actions",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        # Buttons container
        buttons_container = tk.Frame(card_content, bg=self.colors['card_bg'])
        buttons_container.pack(expand=True)

        # Run Azimuthal Integration button
        self.run_btn = tk.Button(buttons_container, text="🐶 Run Integration",
                           command=self.run_integration,
                           bg='#E89FE9', fg='black',
                           font=('Arial', 11, ), relief='flat',
                           width=20, pady=8, cursor='hand2',
                           borderwidth=0, highlightthickness=0)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 25))

        # Interactive Fitting button
        self.fitting_btn = tk.Button(buttons_container, text="✨ Interactive Fitting",
                           command=self.open_interactive_fitting,
                           bg='#9CDCFE', fg='black',
                           font=('Arial', 11, ), relief='flat',
                           width=20, pady=8, cursor='hand2',
                           borderwidth=0, highlightthickness=0)
        self.fitting_btn.pack(side=tk.LEFT)

        # Add hover effects
        def on_enter_run(e):
            self.run_btn.config(bg='#D87FD8')
        def on_leave_run(e):
            self.run_btn.config(bg='#E89FE9')
        def on_enter_fit(e):
            self.fitting_btn.config(bg='#7BC5F0')
        def on_leave_fit(e):
            self.fitting_btn.config(bg='#9CDCFE')

        self.run_btn.bind('<Enter>', on_enter_run)
        self.run_btn.bind('<Leave>', on_leave_run)
        self.fitting_btn.bind('<Enter>', on_enter_fit)
        self.fitting_btn.bind('<Leave>', on_leave_fit)

    def _create_run_button_section(self):
        """Run button directly on background"""
        center_container = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        center_container.pack(fill=tk.X, pady=(0, 10))

        self.run_btn = tk.Button(center_container, text="🌸 Run Azimuthal Integration",
                           command=self.run_integration,
                           bg='#E89FE9', fg='white',
                           font=('Arial', 10, 'bold'), relief='flat',
                           padx=12, pady=5, cursor='hand2')
        self.run_btn.pack()

    def _create_progress_section(self):
        prog_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        prog_frame.pack(fill=tk.X, padx=0, pady=(10, 10))

        self.progress_bar = CuteSheepProgressBar(prog_frame, width=780, height=80)
        self.progress_bar.pack()

    def _create_log_section(self):
        """Log with larger font"""
        log_frame = tk.Frame(self._section_parent(), bg=self.colors['bg'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 20))

        card = self.create_card_frame(log_frame)
        card.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="🧸", bg=self.colors['card_bg'],
                font=('Arial', 16)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 12, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(content, height=12, wrap=tk.WORD,
                                                  font=('Arial', 10),
                                                  bg='#FAFAFA', fg=self.colors['primary'],
                                                  relief='flat', borderwidth=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def update_mode(self):
        # Remove all existing traces before updating UI
        self._remove_all_traces()

        try:
            for widget in self.dynamic_frame.winfo_children():
                widget.destroy()
        except:
            pass

        if self.mode.get() == 'single':
            self._setup_single_sector_ui()
        else:
            self._setup_multiple_sectors_ui()

    def _setup_single_sector_ui(self):
        """Single sector with bin mode option"""
        # Add bin mode toggle
        bin_toggle_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        bin_toggle_frame.pack(fill=tk.X, pady=(5, 10))

        tk.Checkbutton(bin_toggle_frame,
                       text="🍰 Enable Bin Mode",
                       variable=self.bin_mode,
                       bg=self.colors['card_bg'],
                       font=('Arial', 8, 'bold'),
                       command=self._update_bin_mode_ui).pack(anchor=tk.W)

        # Dynamic frame for bin/normal mode
        self.bin_mode_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        self.bin_mode_frame.pack(fill=tk.X, pady=(0, 10))

        self._update_bin_mode_ui()

    def _update_bin_mode_ui(self):
        """Update UI based on bin mode selection"""
        # Remove traces before rebuilding UI
        self._remove_all_traces()

        try:
            for widget in self.bin_mode_frame.winfo_children():
                widget.destroy()
        except:
            pass

        # Process pending events to prevent flicker
        self.bin_mode_frame.update_idletasks()

        if self.bin_mode.get():
            # BIN MODE UI
            bin_container = tk.Frame(self.bin_mode_frame, bg=self.colors['card_bg'])
            bin_container.pack(fill=tk.X)

            # Row 1: Total range
            range_frame = tk.Frame(bin_container, bg=self.colors['card_bg'])
            range_frame.pack(fill=tk.X, pady=(0, 10))

            start_cont = tk.Frame(range_frame, bg=self.colors['card_bg'])
            start_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            tk.Label(start_cont, text="Start (°)",
                    bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'],
                    font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 3))
            tk.Entry(start_cont, textvariable=self.bin_start,
                    font=('Arial', 9), width=15).pack(anchor=tk.W)

            end_cont = tk.Frame(range_frame, bg=self.colors['card_bg'])
            end_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(end_cont, text="End (°)",
                    bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'],
                    font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 3))
            tk.Entry(end_cont, textvariable=self.bin_end,
                    font=('Arial', 9), width=15).pack(anchor=tk.W)

            # Row 2: Bin size and calculated bin count
            step_frame = tk.Frame(bin_container, bg=self.colors['card_bg'])
            step_frame.pack(fill=tk.X)

            step_cont = tk.Frame(step_frame, bg=self.colors['card_bg'])
            step_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            tk.Label(step_cont, text="Bin Size (°)",
                    bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'],
                    font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 3))
            tk.Entry(step_cont, textvariable=self.bin_step,
                    font=('Arial', 9), width=15).pack(anchor=tk.W)

            # Display calculated bin count
            info_cont = tk.Frame(step_frame, bg=self.colors['card_bg'])
            info_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def calculate_bins():
                try:
                    total_range = self.bin_end.get() - self.bin_start.get()
                    step = self.bin_step.get()
                    if step > 0 and total_range > 0:
                        n_bins = int(np.ceil(total_range / step))
                        return f"✨ Will generate {n_bins} bins"
                    return "⚠️ Invalid parameters"
                except:
                    return "⚠️ Invalid parameters"

            self.bin_info_label = tk.Label(info_cont,
                                           text=calculate_bins(),
                                           bg=self.colors['card_bg'],
                                           fg=self.colors['primary'],
                                           font=('Arial', 8, 'italic'))
            self.bin_info_label.pack(anchor=tk.W, pady=(10, 0))

            # Add trace to update bin count in real-time with proper tracking
            def update_bin_info(*args):
                if hasattr(self, 'bin_info_label'):
                    try:
                        self.bin_info_label.config(text=calculate_bins())
                    except:
                        pass

            self._add_trace(self.bin_start, update_bin_info)
            self._add_trace(self.bin_end, update_bin_info)
            self._add_trace(self.bin_step, update_bin_info)

        else:
            # NORMAL SINGLE SECTOR MODE UI
            angle_frame = tk.Frame(self.bin_mode_frame, bg=self.colors['card_bg'])
            angle_frame.pack(fill=tk.X)

            start_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            start_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            tk.Label(start_cont, text="Start (°)", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 3))
            tk.Entry(start_cont, textvariable=self.azimuth_start,
                    font=('Arial', 9), width=12).pack(anchor=tk.W)

            end_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            end_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            tk.Label(end_cont, text="End (°)", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 3))
            tk.Entry(end_cont, textvariable=self.azimuth_end,
                    font=('Arial', 9), width=12).pack(anchor=tk.W)

            label_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            label_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(label_cont, text="Label", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 3))
            tk.Entry(label_cont, textvariable=self.sector_label,
                    font=('Arial', 9), width=15).pack(anchor=tk.W)

    def _setup_multiple_sectors_ui(self):
        """Multiple sectors - Custom sectors only"""
        main_container = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        main_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Directly setup custom sectors mode
        self._setup_custom_sectors_mode_direct(main_container)

    def update_multiple_submode(self):
        # Remove all existing traces before updating UI
        self._remove_all_traces()

        try:
            for widget in self.submode_frame.winfo_children():
                widget.destroy()
            self.sector_row_widgets = []
        except:
            pass

        if self.multiple_mode.get() == 'preset':
            self._setup_preset_mode()
        else:
            self._setup_custom_sectors_mode()

    def _setup_preset_mode(self):
        """Preset mode"""
        preset_frame = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
        preset_frame.pack(fill=tk.X, pady=(5, 0), anchor=tk.W)

        tk.Label(preset_frame, text="Select Preset:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        ttk.Combobox(preset_frame, textvariable=self.preset,
                    values=['quadrants', 'octants', 'hemispheres'],
                    width=28, state='readonly', font=('Arial', 10)).pack(anchor=tk.W)

        preset_info = {
            'quadrants': "4 sectors: 0-90°, 90-180°, 180-270°, 270-360°",
            'octants': "8 sectors: Every 45° from 0° to 360°",
            'hemispheres': "2 sectors: 0-180° (Right), 180-360° (Left)"
        }

        info_text = preset_info.get(self.preset.get(), "Select a preset")
        tk.Label(preset_frame, text=f"🍓 {info_text}",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=(8, 0))

    def _setup_custom_sectors_mode_direct(self, parent_container):
        """Custom sectors with BIN MODE support - Direct setup"""
        if not self.custom_sectors:
            try:
                self.custom_sectors = [
                    [tk.DoubleVar(value=0.0), tk.DoubleVar(value=90.0), tk.StringVar(value="Sector_1"), tk.DoubleVar(value=10.0)],
                    [tk.DoubleVar(value=90.0), tk.DoubleVar(value=180.0), tk.StringVar(value="Sector_2"), tk.DoubleVar(value=10.0)]
                ]
            except Exception as e:
                print(f"Error initializing custom sectors: {e}")
                return

        # Main container
        self.custom_center_all = tk.Frame(parent_container, bg=self.colors['card_bg'])
        self.custom_center_all.pack(expand=True, anchor='center')

        # Bin mode toggle
        bin_toggle_frame = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        bin_toggle_frame.pack(pady=(0, 8), anchor='center')

        tk.Checkbutton(bin_toggle_frame,
                       text="🍰 Enable Bin Mode",
                       variable=self.multi_bin_mode,
                       bg=self.colors['card_bg'],
                       font=('Arial', 8, 'bold'),
                       command=self._update_custom_sectors_display).pack()

        # Sectors container
        sectors_outer_frame = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        sectors_outer_frame.pack(pady=(0, 15), anchor='center')

        self.sectors_spacer = tk.Frame(sectors_outer_frame, bg=self.colors['card_bg'],
                                       height=180, width=1)
        self.sectors_spacer.pack(side=tk.LEFT)

        self.sectors_container = tk.Frame(sectors_outer_frame, bg=self.colors['card_bg'])
        self.sectors_container.pack(side=tk.LEFT, anchor='center')

        # Buttons for add/clear sectors
        btn_frame = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        btn_frame.pack(anchor='center', pady=(10, 0))

        tk.Button(btn_frame, text="🐾 Add Sector", command=self._add_sector,
                 bg='#D8A7D8', fg='white',
                 font=('Arial', 8, 'bold'), relief='flat',
                 padx=5, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="🍉 Clear All", command=self._clear_all_sectors,
                 bg='#FF9FB5', fg='white',
                 font=('Arial', 8, 'bold'), relief='flat',
                 padx=5, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=10)

        for idx in range(len(self.custom_sectors)):
            self._create_sector_row(idx)

    def _update_custom_sectors_display(self):
        """Update instruction text and recreate sector rows when bin mode changes"""
        # Recreate sector rows with visual masking
        if hasattr(self, 'sectors_container'):
            try:
                # 🌸 Create temporary overlay to hide flickering
                overlay = tk.Frame(self.sectors_container, 
                                 bg=self.colors['card_bg'],
                                 width=self.sectors_container.winfo_width(),
                                 height=self.sectors_container.winfo_height())
                overlay.place(x=0, y=0, relwidth=1, relheight=1)
                overlay.lift()
                
                # Update UI synchronously
                self.sectors_container.update_idletasks()
                
                # Destroy all old widgets
                for widget in self.sector_row_widgets:
                    widget.destroy()
                self.sector_row_widgets = []
                
                # Recreate all rows
                for idx in range(len(self.custom_sectors)):
                    self._create_sector_row(idx)
                
                # Force complete redraw
                self.sectors_container.update_idletasks()
                
                # 🌸 Remove overlay after a tiny delay
                self.root.after(50, overlay.destroy)
                
            except Exception as e:
                print(f"Error updating custom sectors display: {e}")

    def _create_sector_row(self, idx):
        """Create a single sector row"""
        try:
            sector = self.custom_sectors[idx]

            if len(sector) == 3:
                sector.append(tk.DoubleVar(value=10.0))
                self.custom_sectors[idx] = sector

            row_frame = tk.Frame(self.sectors_container, bg=self.colors['card_bg'])
            row_frame.pack(pady=3, anchor='center')

            self.sector_row_widgets.append(row_frame)

            num_label = tk.Label(row_frame, text=f"#{idx+1}", bg=self.colors['card_bg'],
                                font=('Arial', 8, 'bold'), width=3)
            num_label.pack(side=tk.LEFT, padx=(0, 6))

            tk.Label(row_frame, text="Start:", bg=self.colors['card_bg'],
                    font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=(0, 3))
            tk.Entry(row_frame, textvariable=sector[0], width=6,
                    font=('Arial', 8), relief='solid', borderwidth=1).pack(side=tk.LEFT, padx=(0, 6))

            tk.Label(row_frame, text="End:", bg=self.colors['card_bg'],
                    font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=(0, 3))
            tk.Entry(row_frame, textvariable=sector[1], width=6,
                    font=('Arial', 8), relief='solid', borderwidth=1).pack(side=tk.LEFT, padx=(0, 6))

            tk.Label(row_frame, text="Label:", bg=self.colors['card_bg'],
                    font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=(0, 3))
            tk.Entry(row_frame, textvariable=sector[2],
                    font=('Arial', 8), width=8, relief='solid', borderwidth=1).pack(side=tk.LEFT, padx=(0, 6))

            if self.multi_bin_mode.get():
                tk.Label(row_frame, text="Bin:", bg=self.colors['card_bg'],
                        font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=(0, 3))
                tk.Entry(row_frame, textvariable=sector[3], width=6,
                        font=('Arial', 8), relief='solid', borderwidth=1).pack(side=tk.LEFT, padx=(0, 6))

                def calculate_sector_bins(s=sector):
                    try:
                        start = s[0].get()
                        end = s[1].get()
                        step = s[3].get()
                        total_range = end - start

                        if step > 0 and total_range > 0:
                            n_bins = int(np.ceil(total_range / step))
                            return f"✨ {n_bins} bins"
                        return "⚠️"
                    except:
                        return "⚠️"

                bin_count_label = tk.Label(row_frame,
                                           text=calculate_sector_bins(),
                                           bg=self.colors['card_bg'],
                                           fg=self.colors['primary'],
                                           font=('Arial', 7, 'italic'),
                                           width=10)
                bin_count_label.pack(side=tk.LEFT, padx=(0, 6))

                def update_bin_count(label=bin_count_label, s=sector):
                    def callback(*args):
                        try:
                            if label.winfo_exists():
                                start = s[0].get()
                                end = s[1].get()
                                step = s[3].get()
                                total_range = end - start
                                if step > 0 and total_range > 0:
                                    n_bins = int(np.ceil(total_range / step))
                                    label.config(text=f"✨ {n_bins} bins")
                                else:
                                    label.config(text="⚠️")
                        except:
                            pass
                    return callback

                callback = update_bin_count()
                self._add_trace(sector[0], callback)
                self._add_trace(sector[1], callback)
                self._add_trace(sector[3], callback)

            tk.Button(row_frame, text="✖", command=lambda i=idx: self._delete_sector(i),
                     bg='#E88C8C', fg='white', font=('Arial', 8, 'bold'),
                     relief='flat', width=2, cursor='hand2').pack(side=tk.LEFT)

        except Exception as e:
            print(f"Error creating sector row {idx}: {e}")

    def _add_sector(self):
        """Add a new sector row"""
        try:
            new_sector = [
                tk.DoubleVar(value=0.0),
                tk.DoubleVar(value=90.0),
                tk.StringVar(value=f"Sector_{len(self.custom_sectors) + 1}"),
                tk.DoubleVar(value=10.0)
            ]
            self.custom_sectors.append(new_sector)
            self._create_sector_row(len(self.custom_sectors) - 1)
        except Exception as e:
            self.log(f"Error adding sector: {e}")

    def _delete_sector(self, index):
        """Delete a sector"""
        if len(self.custom_sectors) <= 1:
            messagebox.showwarning("Warning", "At least one sector must be defined!")
            return

        try:
            self.sectors_container.update_idletasks()
            del self.custom_sectors[index]

            if index < len(self.sector_row_widgets):
                row_widget = self.sector_row_widgets[index]
                row_widget.pack_forget()
                del self.sector_row_widgets[index]
                self._renumber_sectors()
                self.sectors_container.update_idletasks()
                row_widget.destroy()

        except Exception as e:
            self.log(f"Error deleting sector: {e}")

    def _renumber_sectors(self):
        """Update sector number labels"""
        try:
            for idx, row_widget in enumerate(self.sector_row_widgets):
                children = row_widget.winfo_children()
                if children:
                    num_label = children[0]
                    if isinstance(num_label, tk.Label):
                        num_label.config(text=f"#{idx+1}")
        except Exception as e:
            print(f"Error renumbering sectors: {e}")

    def _clear_all_sectors(self):
        """Clear all sectors"""
        result = messagebox.askyesno("Confirm", "Clear all sectors and reset to default?")
        if result:
            try:
                self._remove_all_traces()

                for row_widget in self.sector_row_widgets:
                    row_widget.destroy()
                self.sector_row_widgets = []

                self.custom_sectors = [
                    [tk.DoubleVar(value=0.0),
                     tk.DoubleVar(value=90.0),
                     tk.StringVar(value="Sector_1"),
                     tk.DoubleVar(value=10.0)]
                ]

                self._create_sector_row(0)

            except Exception as e:
                self.log(f"Error clearing sectors: {e}")

    def log(self, message):
        """Thread-safe logging"""
        def _log():
            with self._cleanup_lock:
                if self._is_destroyed:
                    return

            try:
                if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
            except (tk.TclError, RuntimeError):
                pass

        if threading.current_thread() is threading.main_thread():
            _log()
        else:
            try:
                self.root.after(0, _log)
            except (tk.TclError, RuntimeError):
                pass

    def open_interactive_fitting(self):
        """Open the Interactive Peak Fitting GUI from half_auto_fitting.py"""
        try:
            # Create a new Toplevel window instead of a new Tk instance
            fitting_window = tk.Toplevel(self.root)
            fitting_window.withdraw()  # Hide initially

            # Create the PeakFittingGUI instance
            app = PeakFittingGUI(fitting_window)

            # Set custom icon for the fitting window
            if hasattr(app, 'icon_path') and app.icon_path:
                try:
                    fitting_window.iconbitmap(app.icon_path)
                except:
                    pass

            # Show the window after initialization
            fitting_window.deiconify()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Interactive Fitting:\n{str(e)}")
            self.log(f"❌ Error opening Interactive Fitting: {str(e)}")

    def run_integration(self):
        """Run integration with validation"""
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already running!")
            return

        try:
            if not self.poni_path.get():
                messagebox.showerror("Error", "Please select PONI file")
                return
            if not self.input_pattern.get():
                messagebox.showerror("Error", "Please select input H5 files")
                return
            if not self.output_dir.get():
                messagebox.showerror("Error", "Please select output directory")
                return

            dataset_path = self.dataset_path.get().strip()
            if not dataset_path:
                messagebox.showerror("Error", "Dataset path cannot be empty!")
                return

            params = {
                'mode': self.mode.get(),
                'poni_path': self.poni_path.get(),
                'mask_path': self.mask_path.get() if self.mask_path.get() else None,
                'input_pattern': self.input_pattern.get(),
                'output_dir': self.output_dir.get(),
                'dataset_path': dataset_path,
                'npt': self.npt.get(),
                'unit': self.unit.get(),
                'output_csv': self.output_csv.get(),
                'create_stacked_plot': self.create_stacked_plot.get(),
                'stacked_plot_offset': self.stacked_plot_offset.get()
            }

            if params['mode'] == 'single':
                if self.bin_mode.get():
                    bin_start = self.bin_start.get()
                    bin_end = self.bin_end.get()
                    bin_step = self.bin_step.get()

                    if bin_step <= 0:
                        messagebox.showerror("Error", "Bin size must be positive!")
                        return

                    if bin_start >= bin_end:
                        messagebox.showerror("Error", "Start angle must be less than end angle!")
                        return

                    sectors = []
                    current = bin_start
                    bin_idx = 1

                    while current < bin_end:
                        next_angle = min(current + bin_step, bin_end)
                        label = f"Bin{bin_idx:03d}_{current:.1f}-{next_angle:.1f}"
                        sectors.append((float(current), float(next_angle), label))
                        current = next_angle
                        bin_idx += 1

                    params['sectors'] = sectors
                    params['bin_mode'] = True
                else:
                    params['sectors'] = [(
                        float(self.azimuth_start.get()),
                        float(self.azimuth_end.get()),
                        str(self.sector_label.get())
                    )]
                    params['bin_mode'] = False
            else:
                # Multiple sectors - custom sectors only
                if self.multi_bin_mode.get():
                    all_sectors = []
                    for sector_data in self.custom_sectors:
                        start = sector_data[0].get()
                        end = sector_data[1].get()
                        base_label = sector_data[2].get()
                        bin_step = sector_data[3].get()

                        if bin_step <= 0:
                            messagebox.showerror("Error", f"Bin size for {base_label} must be positive!")
                            return

                        current = start
                        bin_idx = 1
                        while current < end:
                            next_angle = min(current + bin_step, end)
                            label = f"{base_label}_Bin{bin_idx:02d}_{current:.1f}-{next_angle:.1f}"
                            all_sectors.append((float(current), float(next_angle), label))
                            current = next_angle
                            bin_idx += 1

                    params['sectors'] = all_sectors
                    params['bin_mode'] = True
                else:
                    sectors = []
                    for sector_data in self.custom_sectors:
                        start = sector_data[0].get()
                        end = sector_data[1].get()
                        label = sector_data[2].get()
                        sectors.append((float(start), float(end), str(label)))
                    params['sectors'] = sectors
                    params['bin_mode'] = False

            self.processing = True
            self.stop_processing = False

            threading.Thread(target=self._run_integration_thread,
                            args=(params,), daemon=True).start()
        except Exception as e:
            self.log(f"Error starting integration: {e}")
            messagebox.showerror("Error", f"Failed to start integration:\n{e}")

    def _run_integration_thread(self, params):
        """Thread-safe integration runner"""
        try:
            def safe_progress_start():
                with self._cleanup_lock:
                    if self._is_destroyed:
                        return
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.start()
                except (tk.TclError, RuntimeError):
                    pass

            # Use try-except to catch errors when calling after from thread
            try:
                self.root.after(0, safe_progress_start)
            except RuntimeError:
                # If main thread is not in main loop, call directly
                safe_progress_start()

            if params['mode'] == 'single':
                if params.get('bin_mode', False):
                    self.log("🍰 Starting Bin Mode Azimuthal Integration")
                else:
                    self.log("🥝 Starting Single Sector Azimuthal Integration")
                self._run_single_sector(params)
            else:
                if params.get('bin_mode', False):
                    self.log("🍰 Starting Multiple Sectors with Bin Mode")
                else:
                    self.log("🍋 Starting Multiple Sectors Azimuthal Integration")
                self._run_multiple_sectors(params)

            if not self.stop_processing:
                self.log("🍇 Azimuthal integration completed!")

                def show_success():
                    with self._cleanup_lock:
                        if self._is_destroyed:
                            return
                    try:
                        messagebox.showinfo("Success", "Azimuthal integration completed successfully!")
                    except:
                        pass

                try:
                    self.root.after(0, show_success)
                except RuntimeError:
                    show_success()

        except Exception as e:
            if not self.stop_processing:
                import traceback
                error_details = traceback.format_exc()
                error_msg = str(e)

                self.log(f"🐤 Error: {error_msg}")
                self.log(f"\nDetails:\n{error_details}")

                def show_error():
                    with self._cleanup_lock:
                        if self._is_destroyed:
                            return
                    try:
                        messagebox.showerror("Error", f"Azimuthal integration failed:\n{error_msg}")
                    except:
                        pass

                try:
                    self.root.after(0, show_error)
                except RuntimeError:
                    show_error()

        finally:
            def safe_progress_stop():
                with self._cleanup_lock:
                    if self._is_destroyed:
                        return
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.stop()
                except (tk.TclError, RuntimeError):
                    pass

            try:
                self.root.after(0, safe_progress_stop)
            except RuntimeError:
                safe_progress_stop()
            self.processing = False

            # Run stacked plot generation after integration is fully finished
            if not self.stop_processing and params.get('create_stacked_plot', False):
                try:
                    self.root.after(0, lambda: self._start_stacked_plot_worker(params.copy()))
                except RuntimeError:
                    self._start_stacked_plot_worker(params.copy())

    def _run_single_sector(self, params):
        """Run single sector integration"""
        self.log(f"🍑 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🧁 Mask file: {os.path.basename(params['mask_path'])}")

        if params.get('bin_mode', False):
            self.log(f"🍰 Bin Mode: Processing {len(params['sectors'])} bins")
            all_output_files = []

            for idx, (azim_start, azim_end, sector_label) in enumerate(params['sectors']):
                if self.stop_processing:
                    self.log("☁️ Processing stopped by user")
                    break

                self.log(f"\n🌈 Bin {idx+1}/{len(params['sectors'])}: {azim_start}° to {azim_end}° ({sector_label})")
                output_files = self._integrate_sector(params, azim_start, azim_end, sector_label)
                all_output_files.extend(output_files)

            if not self.stop_processing:
                self.log(f"\n{'='*60}")
                self.log(f"✨ Bin integration complete!")
                self.log(f"🐨 Generated {len(all_output_files)} files from {len(params['sectors'])} bins")
                self.log(f"🐨 Output directory: {params['output_dir']}")
                self.log(f"{'='*60}\n")
        else:
            azim_start, azim_end, sector_label = params['sectors'][0]
            self.log(f"🌈 Azimuthal range: {azim_start}° to {azim_end}°")
            self.log(f"🌈 Sector label: {sector_label}")

            output_files = self._integrate_sector(params, azim_start, azim_end, sector_label)

            self.log(f"\n{'='*60}")
            self.log(f"✨ Integration complete!")
            self.log(f"🐨 Generated {len(output_files)} files")
            self.log(f"🐨 Output directory: {params['output_dir']}")
            self.log(f"{'='*60}\n")

    def _run_multiple_sectors(self, params):
        """Run multiple sectors integration"""
        self.log(f"🍦 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🍦 Mask file: {os.path.basename(params['mask_path'])}")

        sector_list = params['sectors']

        if params.get('bin_mode', False):
            self.log(f"🦄 Using custom sectors with bin mode")
        else:
            self.log(f"🦄 Using custom sectors")

        self.log(f"🦄 Number of sectors/bins: {len(sector_list)}")

        for start, end, label in sector_list[:5]:
            self.log(f"   - {label}: {start}° to {end}°")
        if len(sector_list) > 5:
            self.log(f"   ... and {len(sector_list) - 5} more")

        all_output_files = []
        for idx, (start, end, label) in enumerate(sector_list):
            if self.stop_processing:
                self.log("☁️ Processing stopped by user")
                break
            self.log(f"\n☁️ [{idx+1}/{len(sector_list)}] Processing {label}...")
            output_files = self._integrate_sector(params, start, end, label)
            all_output_files.extend(output_files)

        if not self.stop_processing:
            self.log(f"\n{'='*60}")
            self.log(f"✨ Integration complete!")
            self.log(f"🐧 Generated {len(all_output_files)} files total")
            self.log(f"🐧 Output directory: {params['output_dir']}")
            self.log(f"{'='*60}\n")

    def _integrate_sector(self, params, azim_start, azim_end, sector_label):
        """Integrate a single sector"""
        integrator = XRDAzimuthalIntegrator(
            params['poni_path'],
            params['mask_path']
        )

        input_files = sorted(glob.glob(params['input_pattern']))
        if not input_files:
            raise ValueError(f"No files found matching pattern: {params['input_pattern']}")

        self.log(f"   Found {len(input_files)} input files")

        output_dir = params['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        csv_data = {}
        output_files = []

        for idx, h5_file in enumerate(input_files):
            if self.stop_processing:
                break

            filename = os.path.basename(h5_file)
            self.log(f"   [{idx+1}/{len(input_files)}] {filename}")

            try:
                output_file, x_data, y_data = integrator.integrate_file(
                    h5_file,
                    output_dir,
                    npt=params['npt'],
                    unit=params['unit'],
                    output_format='xy',
                    azimuth_range=(azim_start, azim_end),
                    sector_label=sector_label,
                    dataset_path=params.get('dataset_path')
                )

                output_files.append(output_file)
                self.log(f"   ✓ Saved: {os.path.basename(output_file)}")

                csv_data[filename] = {
                    'x': x_data,
                    'y': y_data
                }

            except Exception as e:
                self.log(f"   ⚠️ Error processing {filename}: {str(e)}")
                continue

        if params['output_csv'] and csv_data and not self.stop_processing:
            csv_filename = f"azimuthal_integration_{sector_label}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            self._save_csv(csv_data, csv_path, sector_label, params['unit'])
            output_files.append(csv_path)
            self.log(f"   🥥 CSV saved: {csv_filename}")

        return output_files

    def _save_csv(self, csv_data, csv_path, sector_label, unit):
        """Save CSV file"""
        if not csv_data:
            return

        first_key = list(csv_data.keys())[0]
        x_values = csv_data[first_key]['x']

        df_dict = {unit: x_values}

        for filename, data in csv_data.items():
            base_name = os.path.splitext(filename)[0]
            df_dict[base_name] = data['y']

        df = pd.DataFrame(df_dict)
        df.to_csv(csv_path, index=False)

    def _start_stacked_plot_worker(self, params):
        """Start stacked plot generation on the Tk main thread after integration.

        Running the plotting routine on the UI thread avoids Tk/Tcl thread safety
        issues that were observed when creating the plot from a background
        worker. The method still executes asynchronously via ``after`` so the
        integration worker can finish cleanly before plotting begins.
        """

        try:
            self.root.after(0, lambda: self._generate_stacked_plot(params))
        except RuntimeError:
            # Fallback in rare cases where the event loop is not available
            self._generate_stacked_plot(params)

    def _generate_stacked_plot(self, params):
        """Generate stacked plot"""
        try:
            self.log("\n" + "="*60)
            self.log("📊 Generating stacked plot...")

            output_dir = params['output_dir']
            offset = params.get('stacked_plot_offset', 'auto')

            # Create BatchIntegrator instance for generating stacked plot
            # Note: BatchIntegrator requires poni and mask files, but for stacked plot it only reads existing .xy files
            integrator = BatchIntegrator(params['poni_path'], params.get('mask_path'))

            # Call stacked plot generation method
            integrator.create_stacked_plot(
                output_dir=output_dir,
                offset=offset,
                output_name='stacked_plot.png'
            )

            self.log("✓ Stacked plot generation completed!")
            self.log("="*60 + "\n")

        except Exception as e:
            self.log(f"⚠️ Warning: Failed to generate stacked plot: {str(e)}")
            import traceback
            self.log(f"Details:\n{traceback.format_exc()}")

    def _get_preset_sectors(self, preset_name):
        """Get preset sector configurations"""
        presets = {
            'quadrants': [
                (0, 90, "Q1_0-90"),
                (90, 180, "Q2_90-180"),
                (180, 270, "Q3_180-270"),
                (270, 360, "Q4_270-360")
            ],
            'octants': [
                (0, 45, "Oct1_0-45"),
                (45, 90, "Oct2_45-90"),
                (90, 135, "Oct3_90-135"),
                (135, 180, "Oct4_135-180"),
                (180, 225, "Oct5_180-225"),
                (225, 270, "Oct6_225-270"),
                (270, 315, "Oct7_270-315"),
                (315, 360, "Oct8_315-360")
            ],
            'hemispheres': [
                (0, 180, "Right_Hemisphere"),
                (180, 360, "Left_Hemisphere")
            ]
        }
        return presets.get(preset_name, [])


# ============================================================================
# Standalone Entry Point
# ============================================================================

def main():
    """Main entry point with error handling"""
    root = tk.Tk()
    root.title("XRD Azimuthal Integration Tool")
    root.geometry("900x1000")
    root.configure(bg='#F8F3FF')

    main_frame = tk.Frame(root, bg='#F8F3FF')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

    app = AzimuthalIntegrationModule(main_frame, root)
    app.setup_ui()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    finally:
        try:
            root.quit()
        except:
            pass


if __name__ == "__main__":
    main()