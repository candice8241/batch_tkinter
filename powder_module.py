# -*- coding: utf-8 -*-
"""
Powder XRD Module - THREAD-SAFE VERSION WITH IMPROVED UI
Contains integration, peak fitting, phase analysis, and Birch-Murnaghan fitting
FIXED: All thread safety issues + Beautiful success dialogs + Layout improvements
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import threading
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import shutil
import glob
import weakref

# Safely silence Tkinter variable cleanup errors in non-main threads (common in notebooks)
_original_var_del = tk.Variable.__del__


def _safe_var_del(self):
    try:
        if self._tk is not None and self._name:
            exists = self._tk.getboolean(self._tk.call("info", "exists", self._name))
            if exists:
                _original_var_del(self)
    except Exception:
        # Ignore cleanup issues when the Tk main loop has already been destroyed
        pass


tk.Variable.__del__ = _safe_var_del

from batch_integration import BatchIntegrator
from half_auto_fitting import DataProcessor
from batch_cal_volume import XRayDiffractionAnalyzer as XRDAnalyzer
from crysfml_eos_module import CrysFMLEoS, EoSType, MultiEoSFitter
from interactive_eos_gui import InteractiveEoSGUI
from theme_module import GUIBase, CuteSheepProgressBar, ModernTab, ModernButton

# Import the enhanced peak fitting GUI
from half_auto_fitting import PeakFittingGUI


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

        # Bind hover effects
        self.left_btn.bind('<Enter>', lambda e: self.left_btn.config(bg='#F0E6FA'))
        self.left_btn.bind('<Leave>', lambda e: self.left_btn.config(bg='white'))
        self.right_btn.bind('<Enter>', lambda e: self.right_btn.config(bg='#F0E6FA'))
        self.right_btn.bind('<Leave>', lambda e: self.right_btn.config(bg='white'))

        # Bind entry validation
        self.entry.bind('<Return>', self.validate_entry)
        self.entry.bind('<FocusOut>', self.validate_entry)

    def get_current_value(self):
        """Get current value from textvariable"""
        if self.textvariable:
            try:
                if self.is_float:
                    return float(self.textvariable.get())
                else:
                    return int(self.textvariable.get())
            except (ValueError, tk.TclError):
                return self.from_
        return self.from_

    def set_value(self, value):
        """Set value to textvariable"""
        # Clamp value to bounds
        value = max(self.from_, min(self.to, value))

        if self.textvariable:
            if self.is_float:
                self.textvariable.set(float(value))
            else:
                self.textvariable.set(int(value))

    def increase(self):
        """Increase value"""
        current = self.get_current_value()
        new_value = current + self.increment
        self.set_value(new_value)

    def decrease(self):
        """Decrease value"""
        current = self.get_current_value()
        new_value = current - self.increment
        self.set_value(new_value)

    def validate_entry(self, event=None):
        """Validate manual entry"""
        try:
            current = self.get_current_value()
            self.set_value(current)  # This will clamp to bounds
        except:
            self.set_value(self.from_)


class PowderXRDModule(GUIBase):
    """Powder XRD processing module - COMPLETELY THREAD-SAFE VERSION"""

    def __init__(self, parent, root):
        """
        Initialize Powder XRD module

        Args:
            parent: Parent frame to contain this module
            root: Root Tk window for dialogs
        """
        super().__init__()
        self.parent = parent
        self.root = root
        self.current_module = "integration"

        # Use weak references to avoid circular references
        self._root_ref = weakref.ref(root)
        
        # Initialize variables BEFORE any other setup
        self._init_variables()

        # Pre-create module frames
        self.integration_frame = None
        self.analysis_frame = None

        # Track interactive fitting window
        self.interactive_fitting_window = None

        # Track interactive EoS window
        self.interactive_eos_window = None

        # Ensure we only run the expensive prebuild once
        self._interactive_windows_prebuilt = False
        self._interactive_prebuild_job = None

        # Track running threads for cleanup
        self.running_threads = []
        self._is_shutting_down = False
        self._cleanup_lock = threading.Lock()

        # Ensure heavy interactive windows are constructed as soon as the
        # event loop is ready so the first manual open does not rebuild UI
        # on demand (which caused the visible flash).
        self._interactive_prebuild_job = self.root.after_idle(
            self.prebuild_interactive_windows
        )

    def _init_variables(self):
        """Initialize all Tkinter variables - THREAD SAFE with explicit master binding"""
        # Integration and fitting variables
        self.poni_path = tk.StringVar(master=self.root)
        self.mask_path = tk.StringVar(master=self.root)
        self.input_pattern = tk.StringVar(master=self.root)
        self.output_dir = tk.StringVar(master=self.root)
        self.dataset_path = tk.StringVar(master=self.root, value="entry/data/data")
        self.npt = tk.IntVar(master=self.root, value=4000)
        self.unit = tk.StringVar(master=self.root, value='2θ (°)')
        self.fit_method = tk.StringVar(master=self.root, value='pseudo')

        # Output format options (6 formats)
        self.format_xy = tk.BooleanVar(master=self.root, value=True)
        self.format_dat = tk.BooleanVar(master=self.root, value=False)
        self.format_chi = tk.BooleanVar(master=self.root, value=False)
        self.format_fxye = tk.BooleanVar(master=self.root, value=False)
        self.format_svg = tk.BooleanVar(master=self.root, value=False)
        self.format_png = tk.BooleanVar(master=self.root, value=False)

        # Stacked plot options
        self.create_stacked_plot = tk.BooleanVar(master=self.root, value=False)
        self.stacked_plot_offset = tk.StringVar(master=self.root, value='auto')

        # Phase analysis variables
        self.phase_peak_csv = tk.StringVar(master=self.root)
        self.phase_volume_csv = tk.StringVar(master=self.root)
        self.phase_volume_system = tk.StringVar(master=self.root, value='FCC')
        self.phase_volume_output = tk.StringVar(master=self.root)
        self.phase_wavelength = tk.DoubleVar(master=self.root, value=0.4133)
        self.phase_n_points = tk.IntVar(master=self.root, value=4)

        # Birch-Murnaghan / EoS variables
        self.bm_input_file = tk.StringVar(master=self.root)
        self.bm_output_dir = tk.StringVar(master=self.root)
        self.bm_order = tk.StringVar(master=self.root, value='3')
        self.eos_model = tk.StringVar(master=self.root, value='BM-3rd')  # Default: BM 3rd order

    def _start_thread(self, target, name=None):
        """Start a thread and track it for cleanup"""
        if self._is_shutting_down:
            return None

        thread = threading.Thread(target=target, daemon=True, name=name)
        
        with self._cleanup_lock:
            self.running_threads.append(thread)
        
        thread.start()

        # Clean up finished threads from list
        with self._cleanup_lock:
            self.running_threads = [t for t in self.running_threads if t.is_alive()]

        return thread

    def cleanup(self):
        """Clean up resources before shutdown - THREAD-SAFE VERSION"""
        with self._cleanup_lock:
            self._is_shutting_down = True

        # Wait for running threads to complete (with timeout)
        import time
        timeout = 3  # Reduced timeout
        start_time = time.time()

        with self._cleanup_lock:
            threads_to_wait = list(self.running_threads)

        for thread in threads_to_wait:
            if thread.is_alive():
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time > 0:
                    thread.join(timeout=remaining_time)

        # Clear Tkinter variables safely - Set to None instead of deleting
        # This prevents the "main thread is not in main loop" error during cleanup
        try:
            # Store variable names to avoid dict change during iteration
            var_names = [
                'poni_path', 'mask_path', 'input_pattern', 'output_dir',
                'dataset_path', 'npt', 'unit', 'fit_method',
                'format_xy', 'format_dat', 'format_chi', 'format_fxye',
                'format_svg', 'format_png', 'create_stacked_plot', 'stacked_plot_offset',
                'phase_peak_csv', 'phase_volume_csv', 'phase_volume_system',
                'phase_volume_output', 'phase_wavelength', 'phase_n_points',
                'bm_input_file', 'bm_output_dir', 'bm_order'
            ]

            # Set all variables to None instead of deleting them
            # This prevents triggering __del__ on Tkinter variables
            for var_name in var_names:
                if hasattr(self, var_name):
                    try:
                        setattr(self, var_name, None)
                    except:
                        pass
        except:
            pass

    def capture_variables(self):
        """THREAD-SAFE: Capture all Tkinter variables at once"""
        if self._is_shutting_down:
            return None

        try:
            return {
                'poni_path': str(self.poni_path.get()),
                'mask_path': str(self.mask_path.get()),
                'input_pattern': str(self.input_pattern.get()),
                'output_dir': str(self.output_dir.get()),
                'dataset_path': str(self.dataset_path.get()) if self.dataset_path.get() else None,
                'npt': int(self.npt.get()),
                'unit': str(self.unit.get()),
                'fit_method': str(self.fit_method.get()),
                'format_xy': bool(self.format_xy.get()),
                'format_dat': bool(self.format_dat.get()),
                'format_chi': bool(self.format_chi.get()),
                'format_fxye': bool(self.format_fxye.get()),
                'format_svg': bool(self.format_svg.get()),
                'format_png': bool(self.format_png.get()),
                'create_stacked_plot': bool(self.create_stacked_plot.get()),
                'stacked_plot_offset': str(self.stacked_plot_offset.get()),
                'phase_peak_csv': str(self.phase_peak_csv.get()),
                'phase_volume_csv': str(self.phase_volume_csv.get()),
                'phase_volume_system': str(self.phase_volume_system.get()),
                'phase_volume_output': str(self.phase_volume_output.get()),
                'phase_wavelength': float(self.phase_wavelength.get()),
                'phase_n_points': int(self.phase_n_points.get()),
                'bm_input_file': str(self.bm_input_file.get()),
                'bm_output_dir': str(self.bm_output_dir.get()),
                'bm_order': int(self.bm_order.get()),
                'eos_model': str(self.eos_model.get())
            }
        except Exception as e:
            return None

    def setup_ui(self):
        """Setup the complete powder XRD UI"""
        main_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        # Container for all content
        self.dynamic_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        # Build integration and analysis sections inline
        self.integration_frame = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        self.setup_integration_module(self.integration_frame)
        self.integration_frame.pack(fill=tk.BOTH, expand=True)

        self.analysis_frame = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        self.setup_analysis_module(self.analysis_frame)
        self.analysis_frame.pack(fill=tk.BOTH, expand=True)

        # Progress bar section
        prog_cont = tk.Frame(main_frame, bg=self.colors['bg'])
        prog_cont.pack(fill=tk.X, pady=(15, 15))

        prog_inner = tk.Frame(prog_cont, bg=self.colors['bg'])
        prog_inner.pack(expand=True)

        self.progress = CuteSheepProgressBar(prog_inner, width=780, height=80)
        self.progress.pack()

        # Log area
        log_card = self.create_card_frame(main_frame)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_content = tk.Frame(log_card, bg=self.colors['card_bg'], padx=20, pady=12)
        log_content.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_content, bg=self.colors['card_bg'])
        log_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(log_header, text="🐰", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(log_header, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(log_content, height=10, wrap=tk.WORD,
                                                  font=('Arial', 10),
                                                  bg='#FAFAFA', fg='#B794F6',
                                                  relief='flat', borderwidth=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        main_frame.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()

        # Prebuild heavy secondary windows so their first open is immediate
        self.prebuild_interactive_windows()

    def prebuild_interactive_windows(self):
        """Create interactive fitting and EoS windows ahead of user requests."""
        # Cancel any pending scheduled job now that we're running
        if self._interactive_prebuild_job is not None:
            try:
                self.root.after_cancel(self._interactive_prebuild_job)
            except Exception:
                pass
            self._interactive_prebuild_job = None

        if self._interactive_windows_prebuilt:
            return

        # Build both windows while they are fully hidden to avoid any visual flash
        self._build_interactive_fitting_window()
        self._build_interactive_eos_window()
        self._interactive_windows_prebuilt = True

    def _build_interactive_fitting_window(self):
        """Instantiate the interactive fitting UI in a hidden window for reuse."""
        if self.interactive_fitting_window is not None and self.interactive_fitting_window.winfo_exists():
            return self.interactive_fitting_window

        window = tk.Toplevel(self.root)
        window.withdraw()
        window.title("Interactive Peak Fitting - Enhanced")

        window_width = 1400
        window_height = 850
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        icon_paths = [
            "xrd_sheep_icon.ico",
            "xrd_peak_fitting_icon.ico",
            os.path.join(os.path.dirname(__file__), "icon.ico"),
            r"D:\\HEPS\\ID31\\dioptas_data\\github_felicity\\batch\\HP_full_package\\ChatGPT Image.ico"
        ]

        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    window.iconbitmap(icon_path)
                    break
                except Exception:
                    pass

        PeakFittingGUI(window)

        # Fully realize the UI off-screen, then hide it again so first open is instant
        window.attributes("-alpha", 0.0)
        window.deiconify()
        window.update_idletasks()
        window.withdraw()
        window.attributes("-alpha", 1.0)

        def on_closing():
            # Hide without destroying so the taskbar icon does not flash and the
            # widgets stay warm for the next open.
            window.after_idle(window.withdraw)
            window.update_idletasks()
            self.log("📊 Interactive fitting window hidden")

        window.protocol("WM_DELETE_WINDOW", on_closing)
        window.update_idletasks()

        self.interactive_fitting_window = window
        return window

    def _build_interactive_eos_window(self):
        """Instantiate the interactive EoS UI in a hidden window for reuse."""
        if self.interactive_eos_window is not None and self.interactive_eos_window.winfo_exists():
            return self.interactive_eos_window

        window = tk.Toplevel(self.root)
        window.withdraw()
        window.title("Interactive EoS GUI")

        window_width = 1480
        window_height = 900
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        InteractiveEoSGUI(window)

        # Realize layout invisibly so the first visible open uses prebuilt widgets
        window.attributes("-alpha", 0.0)
        window.deiconify()
        window.update_idletasks()
        window.withdraw()
        window.attributes("-alpha", 1.0)

        def on_close():
            try:
                # Hide on idle to avoid a visible flash in the taskbar while
                # keeping the built widgets alive for instant reuse.
                window.after_idle(window.withdraw)
                window.update_idletasks()
                self.log("🌌 Interactive EoS GUI hidden")
            finally:
                self.interactive_eos_window = window

        window.protocol("WM_DELETE_WINDOW", on_close)
        window.update_idletasks()

        self.interactive_eos_window = window
        return window

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

    def setup_integration_module(self, parent_frame):
        """Setup integration and peak fitting module UI - IMPROVED LAYOUT"""
        # ===== MERGED CARD: Integration Settings + Output Formats & Stacked Plot =====
        merged_card = self.create_card_frame(parent_frame)
        merged_card.pack(fill=tk.X, pady=(0, 15))

        content_merged = tk.Frame(merged_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content_merged.pack(fill=tk.BOTH, expand=True)

        # Header for merged card
        header_merged = tk.Frame(content_merged, bg=self.colors['card_bg'])
        header_merged.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header_merged, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header_merged, text="Integration Settings & Output Options",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Container for left-right layout
        main_container = tk.Frame(content_merged, bg=self.colors['card_bg'])
        main_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # ========== LEFT SECTION: Integration Settings (takes more space) ==========
        left_section = tk.Frame(main_container, bg=self.colors['card_bg'])
        left_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 30))

        tk.Label(left_section, text="Integration Settings", bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        self.create_file_picker_with_spinbox_btn(left_section, "PONI File", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(left_section, "Mask File", self.mask_path,
                               [("EDF files", "*.edf"), ("All files", "*.*")])
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

        tk.Radiobutton(unit_options_frame, text="2θ (°)", variable=self.unit, value='2θ (°)',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="Q (Å⁻¹)", variable=self.unit, value='Q (Å⁻¹)',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(unit_options_frame, text="r (mm)", variable=self.unit, value='r (mm)',
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT)

        # ========== RIGHT SECTION: Output Formats & Stacked Plot ==========
        right_outer = tk.Frame(main_container, bg=self.colors['card_bg'], width=350)
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

        # Output Options outer border frame
        output_options_border = tk.Frame(right_section, bg=self.colors['card_bg'],
                                         relief='solid', borderwidth=1)
        output_options_border.pack(fill=tk.X)

        # Output Options content area (with padding)
        output_options_content = tk.Frame(output_options_border, bg=self.colors['card_bg'])
        output_options_content.pack(fill=tk.X, padx=10, pady=10)

        # Right side title (aligned left with content below)
        tk.Label(output_options_content, text="Output Options", bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        # Bottom padding
        tk.Frame(center_container, bg=self.colors['card_bg']).pack(expand=True)

        # Output Formats title (outside border)
        tk.Label(output_options_content, text="Select Output Formats:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 8))

        # Output Formats with border (only frames the six format checkboxes)
        formats_border_frame = tk.Frame(output_options_content, bg=self.colors['card_bg'],
                                       relief='solid', borderwidth=1, highlightbackground='#CCCCCC')
        formats_border_frame.pack(fill=tk.X, pady=(0, 10))

        # Inner padding frame
        formats_content = tk.Frame(formats_border_frame, bg=self.colors['card_bg'])
        formats_content.pack(fill=tk.X, padx=8, pady=8)

        formats_grid = tk.Frame(formats_content, bg=self.colors['card_bg'])
        formats_grid.pack(fill=tk.X)

        # First row of formats (3 checkboxes)
        row1 = tk.Frame(formats_grid, bg=self.colors['card_bg'])
        row1.pack(fill=tk.X, pady=3)

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
        row2 = tk.Frame(formats_grid, bg=self.colors['card_bg'])
        row2.pack(fill=tk.X, pady=3)

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

        # Stacked Plot Options
        tk.Label(output_options_content, text="Stacked Plot Options:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(15, 8))

        # Checkbox on first line
        tk.Checkbutton(output_options_content, text="Create Stacked Plot",
                      variable=self.create_stacked_plot,
                      bg=self.colors['card_bg'], font=('Arial', 9),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(anchor=tk.W, pady=(0, 8))

        # Offset section on second line
        offset_container = tk.Frame(output_options_content, bg=self.colors['card_bg'])
        offset_container.pack(anchor=tk.W, fill=tk.X)

        tk.Label(offset_container, text="Offset:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))

        offset_entry = tk.Entry(offset_container, textvariable=self.stacked_plot_offset,
                               font=('Arial', 10), width=12, justify='center',
                               bg='white', relief='solid', borderwidth=1)
        offset_entry.pack(side=tk.LEFT, ipady=2)

        # Help text below
        tk.Label(output_options_content, text="(use 'auto' or number for offset)",
                bg=self.colors['card_bg'], fg='#888888',
                font=('Arial', 8, 'italic')).pack(anchor=tk.W, pady=(2, 0))

        # ===== Run Integration Button =====
        btn_frame_top = tk.Frame(parent_frame, bg=self.colors['bg'])
        btn_frame_top.pack(fill=tk.X, pady=(10, 15))

        btn_cont_top = tk.Frame(btn_frame_top, bg=self.colors['bg'])
        btn_cont_top.pack(expand=True)

        SpinboxStyleButton(btn_cont_top, "🐿️ Run Integration", self.run_integration,
                          width=200, font_size=11).pack(side=tk.LEFT, padx=(0, 18))

        SpinboxStyleButton(btn_cont_top, "📊 Interactive Fitting", self.open_interactive_fitting,
                          width=200, font_size=11).pack(side=tk.LEFT)

        # Volume Calculation & Lattice Fitting (moved from analysis module)
        volume_card = self.create_card_frame(parent_frame)
        volume_card.pack(fill=tk.X, pady=(0, 15))

        volume_content = tk.Frame(volume_card, bg=self.colors['card_bg'], padx=20, pady=15)
        volume_content.pack(fill=tk.BOTH, expand=True)

        volume_header = tk.Frame(volume_content, bg=self.colors['card_bg'])
        volume_header.pack(anchor=tk.W, pady=(0, 10))

        tk.Label(volume_header, text="🐱", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(volume_header, text="Volume Calculation & Lattice Fitting",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Main container for left content and right section (Crystal System + Wavelength)
        volume_main_container = tk.Frame(volume_content, bg=self.colors['card_bg'])
        volume_main_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # ========== LEFT SECTION: Input fields ==========
        volume_left = tk.Frame(volume_main_container, bg=self.colors['card_bg'])
        volume_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 30))

        tk.Label(volume_left, text="Input CSV (Volume Calculation)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 3))

        volume_input_frame = tk.Frame(volume_left, bg=self.colors['card_bg'])
        volume_input_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Entry(volume_input_frame, textvariable=self.phase_volume_csv, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        SpinboxStyleButton(volume_input_frame, "Browse",
                          lambda: self.browse_file(self.phase_volume_csv, [("CSV files", "*.csv")]),
                          width=75).pack(side=tk.LEFT, padx=(5, 0))

        tk.Label(volume_left, text="Output Directory", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 3))

        volume_output_frame = tk.Frame(volume_left, bg=self.colors['card_bg'])
        volume_output_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Entry(volume_output_frame, textvariable=self.phase_volume_output, font=('Arial', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)

        SpinboxStyleButton(volume_output_frame, "Browse",
                          lambda: self.browse_folder(self.phase_volume_output),
                          width=75).pack(side=tk.LEFT, padx=(5, 0))

        # ========== RIGHT SECTION: Crystal System + Wavelength ==========
        right_outer = tk.Frame(volume_main_container, bg=self.colors['card_bg'], width=350)
        right_outer.pack(side=tk.LEFT, fill=tk.Y, anchor=tk.N)
        right_outer.pack_propagate(False)

        # Create vertical centering container
        center_container = tk.Frame(right_outer, bg=self.colors['card_bg'])
        center_container.pack(fill=tk.BOTH, expand=True)

        # Top padding
        tk.Frame(center_container, bg=self.colors['card_bg']).pack(expand=True)

        # Content area (horizontally and vertically centered)
        right_section = tk.Frame(center_container, bg=self.colors['card_bg'])
        right_section.pack()

        # Outer border frame
        right_border = tk.Frame(right_section, bg=self.colors['card_bg'],
                               relief='solid', borderwidth=1)
        right_border.pack(fill=tk.X, ipady=6)

        # Content area (with padding)
        right_content = tk.Frame(right_border, bg=self.colors['card_bg'])
        right_content.pack(fill=tk.X, padx=8, pady=6)

        # Crystal System section (no title)
        tk.Label(right_content, text="Crystal System:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # Crystal System options (no border wrapper)
        system_content = tk.Frame(right_content, bg=self.colors['card_bg'])
        system_content.pack(fill=tk.X, pady=(0, 8), padx=5)

        # First row: FCC, BCC, Hexagonal, Tetragonal
        system_row1 = tk.Frame(system_content, bg=self.colors['card_bg'])
        system_row1.pack(fill=tk.X, pady=1)

        tk.Radiobutton(system_row1, text="FCC", variable=self.phase_volume_system, value='FCC',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 6))

        tk.Radiobutton(system_row1, text="BCC", variable=self.phase_volume_system, value='BCC',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 6))

        tk.Radiobutton(system_row1, text="Hexagonal", variable=self.phase_volume_system, value='Hexagonal',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 6))

        tk.Radiobutton(system_row1, text="Tetragonal", variable=self.phase_volume_system, value='Tetragonal',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT)

        # Second row: Orthorhombic, Monoclinic, Triclinic
        system_row2 = tk.Frame(system_content, bg=self.colors['card_bg'])
        system_row2.pack(fill=tk.X, pady=1)

        tk.Radiobutton(system_row2, text="Orthorhombic", variable=self.phase_volume_system, value='Orthorhombic',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 6))

        tk.Radiobutton(system_row2, text="Monoclinic", variable=self.phase_volume_system, value='Monoclinic',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT, padx=(0, 6))

        tk.Radiobutton(system_row2, text="Triclinic", variable=self.phase_volume_system, value='Triclinic',
                      bg=self.colors['card_bg'], font=('Arial', 8),
                      fg=self.colors['text_dark'], selectcolor='#E8D5F0',
                      activebackground=self.colors['card_bg']).pack(side=tk.LEFT)

        # Wavelength section
        wavelength_input_frame = tk.Frame(right_content, bg=self.colors['card_bg'])
        wavelength_input_frame.pack(anchor=tk.W, fill=tk.X, pady=(6, 2))

        tk.Label(wavelength_input_frame, text="Wavelength:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 8, 'bold')).pack(side=tk.LEFT)

        wavelength_entry = tk.Entry(wavelength_input_frame, textvariable=self.phase_wavelength,
                                   font=('Arial', 9), width=8, justify='center',
                                   bg='white', relief='solid', borderwidth=1)
        wavelength_entry.pack(side=tk.LEFT, padx=(4, 0), ipady=2)

        tk.Label(wavelength_input_frame, text="Å", bg=self.colors['card_bg'],
                 fg=self.colors['text_dark'], font=('Arial', 8)).pack(side=tk.LEFT, padx=(4, 0))

        # Bottom padding
        tk.Frame(center_container, bg=self.colors['card_bg']).pack(expand=True)

        action_bar = tk.Frame(parent_frame, bg=self.colors['bg'])
        action_bar.pack(fill=tk.X, pady=(0, 10))

        action_container = tk.Frame(action_bar, bg=self.colors['bg'])
        action_container.pack(expand=True)

        SpinboxStyleButton(action_container, "🐼 Calculate Lattice Parameters",
                          self.run_phase_analysis,
                          width=280, font_size=11).pack(side=tk.LEFT, padx=(0, 20))

        SpinboxStyleButton(action_container, "🌌 Open Interactive EoS GUI",
                          self.open_interactive_eos_gui,
                          width=240, font_size=11).pack(side=tk.LEFT)

    def browse_dataset_path(self):
        """Browse for dataset path - FIXED: Using simpledialog instead of creating Toplevel"""
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
            
            if new_path is not None:  # User didn't cancel
                self.dataset_path.set(new_path)

    def open_interactive_fitting(self):
        """Open the interactive peak fitting GUI in a new window"""
        # Guarantee the UI has been built even if the idle prebuild has not run yet
        self.prebuild_interactive_windows()
        window = self._build_interactive_fitting_window()

        try:
            window.deiconify()
            window.lift()
            window.focus_force()
        except Exception:
            pass

        self.log("✨ Interactive peak fitting GUI opened in new window")

    def open_interactive_eos_gui(self):
        """Open the interactive EoS GUI in a separate window"""
        # Guarantee the UI has been built even if the idle prebuild has not run yet
        self.prebuild_interactive_windows()
        window = self._build_interactive_eos_window()

        try:
            window.deiconify()
            window.lift()
            window.focus_force()
        except Exception:
            pass

        self.log("✨ Interactive EoS GUI opened in new window")

    def setup_analysis_module(self, parent_frame):
        """Analysis section currently hosts no additional controls."""
        # Analysis content has been streamlined; EoS controls are available via the
        # dedicated launcher placed alongside the volume calculation action bar.
        placeholder = tk.Frame(parent_frame, bg=self.colors['bg'])
        placeholder.pack(fill=tk.X)

    # ==================== Processing Functions ====================

    def log(self, message):
        """Thread-safe log message - NO CONSOLE OUTPUT"""
        if self._is_shutting_down:
            return

        def _log():
            try:
                if not self._is_shutting_down and hasattr(self, 'log_text'):
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
            except:
                pass

        if threading.current_thread() is threading.main_thread():
            _log()
        else:
            try:
                if not self._is_shutting_down:
                    self.root.after(0, _log)
            except:
                pass

    def show_error(self, title, message):
        """Thread-safe error message"""
        if self._is_shutting_down:
            return

        def _show():
            try:
                if not self._is_shutting_down:
                    messagebox.showerror(title, message)
            except:
                pass

        try:
            if not self._is_shutting_down:
                self.root.after(0, _show)
        except:
            pass

    def show_success_dialog(self, title, message, details=None):
        """Beautiful success dialog with improved UI"""
        if self._is_shutting_down:
            return

        def _show():
            if self._is_shutting_down:
                return
            dialog = tk.Toplevel(self.root)
            dialog.title(title)
            dialog.configure(bg='#F4ECFF')
            dialog.transient(self.root)
            dialog.grab_set()

            # Calculate size based on content
            width = 440
            height = 260 if details else 210

            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            dialog.geometry(f"{width}x{height}+{x}+{y}")

            # Decorative top bar
            top_bar = tk.Frame(dialog, bg='#CDB5F6', height=8)
            top_bar.pack(fill=tk.X)

            # Success icon with soft checkbox look
            icon_frame = tk.Frame(dialog, bg='#F4ECFF')
            icon_frame.pack(pady=14)

            checkbox = tk.Frame(
                icon_frame,
                bg='#F7F2FF',
                relief='solid',
                borderwidth=2,
                highlightbackground='#CDB5F6',
                highlightthickness=2,
                padx=12,
                pady=8
            )
            checkbox.pack()

            tk.Label(
                checkbox,
                text="✓",
                bg='#F7F2FF',
                fg='#8A66D9',
                font=('Arial', 30, 'bold')
            ).pack()

            # Title with shadow effect
            title_frame = tk.Frame(dialog, bg='#F4ECFF')
            title_frame.pack(pady=4)

            tk.Label(title_frame, text=title, bg='#F4ECFF',
                    fg='#8A66D9', font=('Arial', 15, 'bold')).pack()

            # Message card with border
            msg_card = tk.Frame(dialog, bg='#F9F5FF', relief='solid',
                               borderwidth=2, highlightbackground='#CDB5F6',
                               highlightthickness=2)
            msg_card.pack(padx=26, pady=12, fill=tk.BOTH, expand=True)

            msg_inner = tk.Frame(msg_card, bg='#F9F5FF', padx=18, pady=12)
            msg_inner.pack(fill=tk.BOTH, expand=True)

            tk.Label(msg_inner, text=message, bg='#F9F5FF',
                    fg='#333333', font=('Arial', 11),
                    wraplength=360, justify=tk.LEFT).pack()

            if details:
                tk.Frame(msg_inner, bg='#E3D7FF', height=1).pack(fill=tk.X, pady=8)
                tk.Label(msg_inner, text=details, bg='#F9F5FF',
                        fg='#555555', font=('Arial', 9),
                        wraplength=360, justify=tk.LEFT).pack()

            # OK Button with gradient-like effect
            btn_frame = tk.Frame(dialog, bg='#F4ECFF')
            btn_frame.pack(pady=12)
            
            ok_btn = tk.Button(
                btn_frame,
                text="🎉 Awesome!",
                command=dialog.destroy,
                bg='#C8A2D9',
                fg='white',
                font=('Arial', 12, 'bold'),
                relief='flat',
                borderwidth=0,
                padx=40,
                pady=12,
                cursor='hand2'
            )
            ok_btn.pack()
            
            # Enhanced hover effect
            def on_enter(e):
                ok_btn.config(bg='#B794F6', font=('Arial', 12, 'bold'))
            def on_leave(e):
                ok_btn.config(bg='#C8A2D9', font=('Arial', 12, 'bold'))
            
            ok_btn.bind('<Enter>', on_enter)
            ok_btn.bind('<Leave>', on_leave)
            
            dialog.bind('<Return>', lambda e: dialog.destroy())
            ok_btn.focus()

        try:
            if not self._is_shutting_down:
                self.root.after(0, _show)
        except:
            pass

    def browse_file(self, var, filetypes):
        """Browse for file"""
        try:
            filename = filedialog.askopenfilename(filetypes=filetypes)
            if filename:
                var.set(filename)
        except Exception as e:
            self.log(f"❌ Error browsing file: {str(e)}")

    def browse_folder(self, var):
        """Browse for folder"""
        try:
            foldername = filedialog.askdirectory()
            if foldername:
                var.set(foldername)
        except Exception as e:
            self.log(f"❌ Error browsing folder: {str(e)}")

    def separate_peaks(self):
        """Separate original and new peaks from input CSV"""
        if not self.phase_peak_csv.get():
            self.show_error("Error", "Please select peak CSV file first")
            return
        self._start_thread(self._separate_peaks_thread, name="SeparatePeaks")

    def _separate_peaks_thread(self):
        """Background thread for peak separation - THREAD SAFE"""
        vars = self.capture_variables()
        if vars is None:
            self.log("❌ Failed to read settings")
            return

        try:
            self.root.after(0, self.progress.start)
            self.log("🔀 Starting peak separation process...")

            analyzer = XRDAnalyzer(
                wavelength=vars['phase_wavelength'],
                n_pressure_points=vars['phase_n_points']
            )

            self.log(f"📄 Reading data from: {os.path.basename(vars['phase_peak_csv'])}")
            pressure_data = analyzer.read_pressure_peak_data(vars['phase_peak_csv'])
            self.log(f"✓ Loaded {len(pressure_data)} pressure points")

            self.log("🔍 Identifying phase transition...")
            transition_pressure, before_pressures, after_pressures = analyzer.find_phase_transition_point()

            if transition_pressure is None:
                self.log("⚠️ No phase transition detected")
                msg = "No phase transition detected in the data"
                try:
                    self.root.after(0, lambda: messagebox.showwarning("Warning", msg))
                except:
                    pass
                return

            self.log(f"✓ Phase transition detected at {transition_pressure:.2f} GPa")

            transition_peaks = pressure_data[transition_pressure]
            prev_pressure = before_pressures[-1]
            prev_peaks = pressure_data[prev_pressure]

            tolerance_windows = [(p - analyzer.peak_tolerance_1, p + analyzer.peak_tolerance_1)
                                for p in prev_peaks]
            new_peaks_at_transition = []

            for peak in transition_peaks:
                in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
                if not in_any_window:
                    new_peaks_at_transition.append(peak)

            self.log(f"✓ Found {len(new_peaks_at_transition)} new peaks at transition")

            base_filename = vars['phase_peak_csv'].replace('.csv', '')
            new_peaks_dataset_csv = f"{base_filename}_new_peaks_dataset.csv"
            original_peaks_dataset_csv = f"{base_filename}_original_peaks_dataset.csv"

            self.log("📊 Tracking new peaks across pressure points...")
            stable_count, tracked_new_peaks = analyzer.collect_tracked_new_peaks(
                pressure_data,
                transition_pressure,
                after_pressures,
                new_peaks_at_transition,
                analyzer.peak_tolerance_2,
                output_csv=new_peaks_dataset_csv
            )

            self.log(f"✓ {stable_count} stable new peaks identified")
            self.log(f"💾 New peaks dataset saved to: {os.path.basename(new_peaks_dataset_csv)}")

            self.log("📊 Building original peaks dataset...")
            original_peak_dataset = analyzer.build_original_peak_dataset(
                pressure_data,
                tracked_new_peaks,
                analyzer.peak_tolerance_3,
                output_csv=original_peaks_dataset_csv
            )

            self.log(f"✓ Original peaks dataset constructed for {len(original_peak_dataset)} pressure points")
            self.log(f"💾 Original peaks dataset saved to: {os.path.basename(original_peaks_dataset_csv)}")

            self.log("\n" + "="*60)
            self.log("✅ Peak separation completed successfully!")
            self.log("="*60)
            self.log(f"📍 Transition pressure: {transition_pressure:.2f} GPa")
            self.log(f"📊 New peaks CSV: {os.path.basename(new_peaks_dataset_csv)}")
            self.log(f"📊 Original peaks CSV: {os.path.basename(original_peaks_dataset_csv)}")
            self.log("="*60 + "\n")

            msg = "Peak separation completed successfully!"
            details = f"Transition at {transition_pressure:.2f} GPa\nFiles saved to input directory"
            self.show_success_dialog("Success", msg, details)

        except Exception as e:
            error_msg = str(e)
            self.log(f"❌ Error during peak separation: {error_msg}")
            try:
                self.root.after(0, lambda m=error_msg: self.show_error("Error", f"Peak separation failed:\n{m}"))
            except:
                pass
        finally:
            try:
                self.root.after(0, self.progress.stop)
            except:
                pass

    def run_integration(self):
        """Run 1D integration"""
        if not self.poni_path.get() or not self.mask_path.get() or not self.input_pattern.get() or not self.output_dir.get():
            self.show_error("Error", "Please fill all required fields")
            return
        self._start_thread(self._run_integration_thread, name="Integration")

    def _run_integration_thread(self):
        """Background thread for integration - THREAD SAFE"""
        vars = self.capture_variables()
        if vars is None:
            self.log("❌ Failed to read settings")
            return

        formats = []
        if vars['format_xy']: formats.append('xy')
        if vars['format_dat']: formats.append('dat')
        if vars['format_chi']: formats.append('chi')
        if vars['format_fxye']: formats.append('fxye')
        if vars['format_svg']: formats.append('svg')
        if vars['format_png']: formats.append('png')
        if not formats: formats = ['xy']

        # Convert UI-friendly unit to pyFAI-compatible unit
        unit_conversion = {
            '2θ (°)': '2th_deg',
            'Q (Å⁻¹)': 'q_A^-1',
            'r (mm)': 'r_mm'
        }
        pyfai_unit = unit_conversion.get(vars['unit'], vars['unit'])

        try:
            self.root.after(0, self.progress.start)

            if os.path.isdir(vars['input_pattern']):
                target_dir = vars['input_pattern']
            elif os.path.isfile(vars['input_pattern']) and vars['input_pattern'].lower().endswith('.h5'):
                target_dir = os.path.dirname(vars['input_pattern'])
            else:
                raise ValueError(f"Invalid input: {vars['input_pattern']}")

            h5_files = sorted([os.path.join(target_dir, f)
                              for f in os.listdir(target_dir)
                              if f.lower().endswith('.h5')])

            if not h5_files:
                raise ValueError(f"No .h5 files found in directory: {target_dir}")

            total_files = len(h5_files)
            self.log(f"\n{'='*60}")
            self.log(f"🔁 Starting Batch Integration")
            self.log(f"📁 Directory: {target_dir}")
            self.log(f"📊 Total files to process: {total_files}")
            self.log(f"📈 Output formats: {', '.join(formats)}")
            self.log(f"📉 Number of points: {vars['npt']}")
            self.log(f"📏 Unit: {vars['unit']} (pyFAI: {pyfai_unit})")
            self.log(f"{'='*60}\n")

            integrator = BatchIntegrator(vars['poni_path'], vars['mask_path'])

            for i, h5_file in enumerate(h5_files, 1):
                self.log(f"[{i}/{total_files}] Processing: {os.path.basename(h5_file)}")

                integrator.batch_integrate(
                    input_pattern=h5_file,
                    output_dir=vars['output_dir'],
                    npt=vars['npt'],
                    unit=pyfai_unit,
                    dataset_path=vars['dataset_path'],
                    formats=formats,
                    create_stacked_plot=False
                )

                self.log(f"[{i}/{total_files}] ✓ Completed: {os.path.basename(h5_file)}\n")

            if vars['create_stacked_plot'] and total_files > 1:
                self.log(f"📈 Creating combined stacked plot for all {total_files} files...")
                self._create_combined_stacked_plot(vars['output_dir'], vars['stacked_plot_offset'], pyfai_unit)

            self.log(f"\n{'='*60}")
            self.log(f"✅ All integrations completed!")
            self.log(f"📊 Total processed: {total_files}/{total_files}")
            self.log(f"💾 Output directory: {vars['output_dir']}")
            self.log(f"{'='*60}\n")

            msg = "Integration completed successfully!"
            details = f"{total_files} file(s) processed"
            self.show_success_dialog("Integration Complete", msg, details)

        except Exception as e:
            error_msg = str(e)
            self.log(f"❌ Error: {error_msg}")
            try:
                self.root.after(0, lambda msg=error_msg: self.show_error("Error", msg))
            except:
                pass
        finally:
            try:
                self.root.after(0, self.progress.stop)
            except:
                pass

    def _extract_pressure_from_filename(self, filename):
        """
        Extract pressure value from filename

        Supports both loading and unloading data:
        - Loading: 10, 10.5, 40, etc.
        - Unloading: d3.21, d38.2, etc. (prefix 'd' indicates unloading)

        Returns:
            tuple: (pressure, is_unload) where is_unload is True if filename starts with 'd'
        """
        import re
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]

        is_unload = False

        # Check if filename starts with 'd' (unloading data)
        if name_without_ext.startswith('d') or name_without_ext.startswith('D'):
            is_unload = True
            # Remove the 'd' prefix for pressure extraction
            name_without_ext = name_without_ext[1:]

        # Try to extract pressure value
        matches = re.findall(r'(\d+\.?\d*)\s*(?:GPa|gpa|_GPa|_gpa)?', name_without_ext)
        if matches:
            try:
                return float(matches[0]), is_unload
            except:
                pass
        return 0.0, is_unload

    def _create_combined_stacked_plot(self, output_dir, offset, unit='2th_deg'):
        """Create a stacked plot combining all integrated files, sorted by pressure - FIXED LABEL POSITIONING"""
        # Map unit to axis label
        unit_labels = {
            '2th_deg': '2θ (°)',
            'q_A^-1': 'Q (Å⁻¹)',
            'r_mm': 'r (mm)'
        }
        xlabel = unit_labels.get(unit, unit)

        try:
            xy_files = glob.glob(os.path.join(output_dir, "*.xy"))

            if not xy_files:
                self.log("⚠️ No .xy files found for stacked plot")
                return

            # Extract pressure and is_unload info
            file_info_list = []
            for xy_file in xy_files:
                pressure, is_unload = self._extract_pressure_from_filename(xy_file)
                file_info_list.append((xy_file, pressure, is_unload))

            # Separate loading and unloading data
            loading_data = [(f, p) for f, p, u in file_info_list if not u]
            unloading_data = [(f, p) for f, p, u in file_info_list if u]

            # Sort loading data: low to high pressure
            loading_data.sort(key=lambda x: x[1])

            # Sort unloading data: high to low pressure
            unloading_data.sort(key=lambda x: x[1], reverse=True)

            # Combine: loading first, then unloading
            file_pressure_pairs = loading_data + unloading_data

            xy_files_sorted = [fp[0] for fp in file_pressure_pairs]
            pressures = [fp[1] for fp in file_pressure_pairs]
            is_unload_list = [self._extract_pressure_from_filename(f)[1] for f in xy_files_sorted]

            self.log(f"📊 Sorted {len(xy_files_sorted)} files: {len(loading_data)} loading + {len(unloading_data)} unloading")
            self.log(f"📈 Pressure order: {pressures}")

            fig, ax = plt.subplots(figsize=(12, 10))

            # Calculate offset value
            if offset == 'auto':
                max_intensities = []
                for xy_file in xy_files_sorted:
                    data = np.loadtxt(xy_file)
                    max_intensities.append(np.max(data[:, 1]))
                offset_value = np.mean(max_intensities) * 0.5
            else:
                try:
                    offset_value = float(offset)
                except:
                    offset_value = 500  # Default fallback

            all_x_min = float('inf')
            all_x_max = float('-inf')

            for xy_file in xy_files_sorted:
                data = np.loadtxt(xy_file)
                all_x_min = min(all_x_min, np.min(data[:, 0]))
                all_x_max = max(all_x_max, np.max(data[:, 0]))

            x_min = np.ceil(all_x_min)
            x_max = np.floor(all_x_max)

            color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            for i, (xy_file, pressure, is_unload) in enumerate(zip(xy_files_sorted, pressures, is_unload_list)):
                data = np.loadtxt(xy_file)
                x, y = data[:, 0], data[:, 1]
                y_offset = y + i * offset_value

                color_idx = int(pressure / 10) % len(color_palette)
                curve_color = color_palette[color_idx]

                ax.plot(x, y_offset, linewidth=2, alpha=0.8, color=curve_color)

                # FIXED: Position label from the lowest pressure curve (baseline + min intensity)
                # For the first curve (i=0), the label starts from above the baseline
                # For subsequent curves, labels are positioned at their respective baselines
                baseline_y = i * offset_value

                # Find the minimum intensity value at the label position to place label above the curve
                label_x_pos = x_min + 0.03 * (x_max - x_min)
                # Find y value at label x position (approximate using nearest point)
                idx = np.argmin(np.abs(x - label_x_pos))
                label_y = y_offset[idx] if idx < len(y_offset) else baseline_y

                # Add 'd' prefix to label if it's unloading data
                label_text = f'd{pressure:.1f} GPa' if is_unload else f'{pressure:.1f} GPa'

                ax.text(label_x_pos,
                       label_y,
                       label_text,
                       fontsize=9,
                       verticalalignment='bottom',  # Changed to 'bottom' so label sits above the curve
                       horizontalalignment='left',
                       #bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
                       )

            ax.set_xlim(x_min, x_max)
            #ax.text(0.02, 0.78, 'P (GPa)',
            #transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
            ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
            ax.set_ylabel('Intensity (offset)', fontsize=13, fontweight='bold')
            ax.set_title('Stacked XRD Patterns (Loading + Unloading)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            plt.tight_layout()

            stacked_plot_path = os.path.join(output_dir, 'combined_stacked_plot.png')
            plt.savefig(stacked_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            self.log(f"💾 Combined stacked plot saved: {os.path.basename(stacked_plot_path)}")
            self.log(f"📈 Total patterns: {len(xy_files_sorted)}")
            self.log(f"📤 Loading data: {len(loading_data)}, Unloading data: {len(unloading_data)}")
            self.log(f"📊 Pressure range: {min(pressures):.1f} - {max(pressures):.1f} GPa")
            self.log(f"📏 Offset value used: {offset_value:.2f}")
            self.log(f"🎨 Colors change every 10 GPa")

        except Exception as e:
            error_msg = f"⚠️ Failed to create combined stacked plot: {str(e)}"
            self.log(error_msg)

    def run_fitting(self):
        """Run peak fitting"""
        if not self.output_dir.get():
            self.show_error("Error", "Please specify output directory")
            return
        self._start_thread(self._run_fitting_thread, name="Fitting")

    def _run_fitting_thread(self):
        """Background thread for peak fitting - THREAD SAFE"""
        vars = self.capture_variables()
        if vars is None:
            self.log("❌ Failed to read settings")
            return

        try:
            self.root.after(0, self.progress.start)
            self.log("📈 Starting Batch Fitting")

            fitter = DataProcessor(folder=vars['output_dir'], fit_method=vars['fit_method'])
            fitter.run_batch_fitting()

            self.log("✅ Fitting completed!")
            
            msg = "Peak fitting completed successfully!"
            details = f"Method: {vars['fit_method']}"
            self.show_success_dialog("Fitting Complete", msg, details)

        except Exception as e:
            error_msg = str(e)
            self.log(f"❌ Error: {error_msg}")
            try:
                self.root.after(0, lambda msg=error_msg: self.show_error("Error", msg))
            except:
                pass
        finally:
            try:
                self.root.after(0, self.progress.stop)
            except:
                pass

    def run_phase_analysis(self):
        """Run volume calculation and lattice parameter fitting"""
        if not self.phase_volume_csv.get() or not self.phase_volume_output.get():
            self.show_error("Error", "Please fill all required fields (Input CSV and Output Directory)")
            return
        self._start_thread(self._run_phase_analysis_thread, name="PhaseAnalysis")

    def _run_phase_analysis_thread(self):
        """Background thread for phase analysis - THREAD SAFE"""
        vars = self.capture_variables()
        if vars is None:
            self.log("❌ Failed to read settings")
            return

        try:
            self.root.after(0, self.progress.start)
            self.log("🐶 Starting Volume Calculation & Lattice Parameter Fitting")

            os.makedirs(vars['phase_volume_output'], exist_ok=True)

            system_mapping = {
                'FCC': 'cubic_FCC',
                'BCC': 'cubic_BCC',
                'SC': 'cubic_SC',
                'Hexagonal': 'Hexagonal',
                'Tetragonal': 'Tetragonal',
                'Orthorhombic': 'Orthorhombic',
                'Monoclinic': 'Monoclinic',
                'Triclinic': 'Triclinic'
            }

            crystal_system = system_mapping.get(vars['phase_volume_system'], 'cubic_FCC')

            self.log(f"📄 Input CSV: {os.path.basename(vars['phase_volume_csv'])}")
            self.log(f"🔷 Crystal system: {vars['phase_volume_system']}")
            self.log(f"📏 Wavelength: {vars['phase_wavelength']} Å")
            self.log(f"📁 Output directory: {vars['phase_volume_output']}")

            analyzer = XRDAnalyzer(
                wavelength=vars['phase_wavelength'],
                n_pressure_points=vars['phase_n_points']
            )

            self.log("\n" + "="*60)
            self.log("Starting analysis...")
            self.log("="*60 + "\n")

            results = analyzer.analyze(
                csv_path=vars['phase_volume_csv'],
                original_system=crystal_system,
                new_system=crystal_system,
                auto_mode=True
            )

            if results is None:
                self.log("❌ Analysis failed - no results returned")
                self.show_error("Error", "Analysis failed to complete")
                return

            input_dir = os.path.dirname(vars['phase_volume_csv'])
            base_filename = os.path.splitext(os.path.basename(vars['phase_volume_csv']))[0]

            generated_files = []

            possible_files = [
                f"{base_filename}_original_peaks_lattice.csv",
                f"{base_filename}_new_peaks_lattice.csv",
                f"{base_filename}_lattice_results.csv",
                f"{base_filename}_new_peaks_dataset.csv",
                f"{base_filename}_original_peaks_dataset.csv"
            ]

            for filename in possible_files:
                source_path = os.path.join(input_dir, filename)
                if os.path.exists(source_path):
                    dest_path = os.path.join(vars['phase_volume_output'], filename)
                    shutil.copy2(source_path, dest_path)
                    generated_files.append(filename)
                    self.log(f"📋 Copied: {filename}")

            self.log("\n" + "="*60)
            self.log("✅ Volume Calculation & Lattice Fitting Completed!")
            self.log("="*60)

            if 'transition_pressure' in results:
                self.log(f"📍 Phase transition pressure: {results['transition_pressure']:.2f} GPa")

            self.log(f"📁 Output location: {vars['phase_volume_output']}")
            self.log(f"📊 Generated {len(generated_files)} result file(s)")

            for f in generated_files:
                self.log(f"   - {f}")

            self.log("="*60 + "\n")

            msg = "Volume calculation completed successfully!"
            details = ""
            if 'transition_pressure' in results:
                details += f"Transition at {results['transition_pressure']:.2f} GPa\n"
            details += f"{len(generated_files)} file(s) saved"
            
            self.show_success_dialog("Success", msg, details)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = str(e)
            self.log(f"❌ Error during analysis: {error_msg}")
            self.log(f"Details:\n{error_details}")
            self.show_error("Error", f"Volume calculation failed:\n{error_msg}")
        finally:
            self.root.after(0, self.progress.stop)

    def run_birch_murnaghan(self):
        """Run Birch-Murnaghan equation of state fitting"""
        if not self.bm_input_file.get() or not self.bm_output_dir.get():
            self.show_error("Error", "Please fill all required fields")
            return
        self._start_thread(self._run_birch_murnaghan_thread, name="BirchMurnaghan")

    def _run_birch_murnaghan_thread(self):
        """Background thread for EoS fitting using CrysFML module - THREAD SAFE"""
        vars = self.capture_variables()
        if vars is None:
            self.log("❌ Failed to read settings")
            return

        try:
            self.root.after(0, self.progress.start)

            # Get EoS model selection
            eos_model_name = vars.get('eos_model', 'BM-3rd')

            # Map UI model names to EoSType and display names
            model_map = {
                'BM-2nd': (EoSType.BIRCH_MURNAGHAN_2ND, '2nd order Birch-Murnaghan'),
                'BM-3rd': (EoSType.BIRCH_MURNAGHAN_3RD, '3rd order Birch-Murnaghan'),
                'BM-4th': (EoSType.BIRCH_MURNAGHAN_4TH, '4th order Birch-Murnaghan'),
                'Murnaghan': (EoSType.MURNAGHAN, 'Murnaghan'),
                'Vinet': (EoSType.VINET, 'Vinet'),
                'Natural Strain': (EoSType.NATURAL_STRAIN, 'Natural Strain'),
                'Tait': (EoSType.TAIT, 'Tait')
            }

            # Determine EoS type and display name
            eos_type, model_display = model_map.get(eos_model_name, (EoSType.BIRCH_MURNAGHAN_3RD, '3rd order Birch-Murnaghan'))

            self.log(f"⚗️ Starting {model_display} EoS Fitting")

            os.makedirs(vars['bm_output_dir'], exist_ok=True)

            self.log(f"📄 Reading data from: {os.path.basename(vars['bm_input_file'])}")
            df = pd.read_csv(vars['bm_input_file'])

            if 'V_atomic' not in df.columns or 'Pressure (GPa)' not in df.columns:
                raise ValueError("CSV must contain 'V_atomic' and 'Pressure (GPa)' columns")

            V_data = df['V_atomic'].dropna().values
            P_data = df['Pressure (GPa)'].dropna().values

            min_len = min(len(V_data), len(P_data))
            V_data = V_data[:min_len]
            P_data = P_data[:min_len]

            self.log(f"✓ Loaded {len(V_data)} data points")
            self.log(f"   Volume range: {V_data.min():.4f} - {V_data.max():.4f} Å³/atom")
            self.log(f"   Pressure range: {P_data.min():.2f} - {P_data.max():.2f} GPa")

            # Create CrysFML EoS fitter
            self.log(f"\n🔧 Fitting {model_display} equation...")
            fitter = CrysFMLEoS(
                eos_type=eos_type,
                V0_bounds=(0.8, 1.3),
                B0_bounds=(50, 500),
                B0_prime_bounds=(2.5, 6.5),
                max_iterations=10000
            )

            # Fit the data
            params = fitter.fit(V_data, P_data)

            if params is None:
                raise ValueError(f"{model_display} fitting failed")

            # Log results
            self.log(f"\n{'='*60}")
            self.log(f"✅ {model_display} EoS Fitting Results:")
            self.log(f"{'='*60}")
            self.log(f"📊 V₀ = {params.V0:.4f} ± {params.V0_err:.4f} Å³/atom")
            self.log(f"📊 B₀ = {params.B0:.2f} ± {params.B0_err:.2f} GPa")
            self.log(f"📊 B₀' = {params.B0_prime:.3f} ± {params.B0_prime_err:.3f}")
            if abs(params.B0_prime2) > 1e-10:
                self.log(f"📊 B₀'' = {params.B0_prime2:.4f} ± {params.B0_prime2_err:.4f} GPa⁻¹")
            self.log(f"📈 R² = {params.R_squared:.6f}")
            self.log(f"📉 RMSE = {params.RMSE:.4f} GPa")
            self.log(f"{'='*60}\n")

            # Generate plots
            self.log("📈 Generating plots...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            ax1.scatter(V_data, P_data, s=80, c='blue', marker='o',
                       label='Experimental Data', alpha=0.7, edgecolors='black', linewidths=1.5)

            V_fit = np.linspace(V_data.min()*0.95, V_data.max()*1.05, 200)
            P_fit = fitter.calculate_pressure(V_fit, params)
            P_data_fit = fitter.calculate_pressure(V_data, params)

            # Color selection based on model
            color_map = {
                EoSType.BIRCH_MURNAGHAN_2ND: 'red',
                EoSType.BIRCH_MURNAGHAN_3RD: 'green',
                EoSType.BIRCH_MURNAGHAN_4TH: 'purple',
                EoSType.MURNAGHAN: 'orange',
                EoSType.VINET: 'blue',
                EoSType.NATURAL_STRAIN: 'brown',
                EoSType.TAIT: 'magenta'
            }
            color = color_map.get(eos_type, 'green')

            ax1.plot(V_fit, P_fit, color=color, linewidth=2.5,
                    label=f'{model_display} Fit', alpha=0.8)
            ax1.set_xlabel('Volume V (Å³/atom)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Pressure P (GPa)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{model_display} Equation of State',
                         fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Build parameter text
            textstr = f"$V_0$ = {params.V0:.4f} ± {params.V0_err:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {params.B0:.2f} ± {params.B0_err:.2f} GPa\n"
            textstr += f"$B_0'$ = {params.B0_prime:.3f} ± {params.B0_prime_err:.3f}\n"
            if abs(params.B0_prime2) > 1e-10:
                textstr += f"$B_0''$ = {params.B0_prime2:.4f} GPa⁻¹\n"
            textstr += f"$R^2$ = {params.R_squared:.6f}\n"
            textstr += f"RMSE = {params.RMSE:.4f} GPa"

            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.8, edgecolor='black'))

            # Residuals plot
            residuals = P_data - P_data_fit
            ax2.scatter(V_data, residuals, s=60, c='blue', marker='o',
                       alpha=0.7, edgecolors='black', linewidths=1.5)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
            ax2.set_xlabel('Volume V (Å³/atom)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals (GPa)', fontsize=12, fontweight='bold')
            ax2.set_title('Fitting Residuals Analysis', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=10)

            rmse_text = f"RMSE = {params.RMSE:.4f} GPa"
            ax2.text(0.05, 0.95, rmse_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightblue', alpha=0.8, edgecolor='black'))

            plt.tight_layout()

            # Save figure
            model_filename = eos_model_name.replace(' ', '_').replace('-', '_').lower()
            fig_path = os.path.join(vars['bm_output_dir'], f'EoS_{model_filename}_fit.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.log(f"💾 Plot saved: {os.path.basename(fig_path)}")

            self.log(f"\n{'='*60}")
            self.log("✨ All tasks completed successfully!")
            self.log(f"{'='*60}")
            self.log(f"📁 Output directory: {vars['bm_output_dir']}")
            self.log(f"   - {os.path.basename(fig_path)} : P-V curve and residuals")
            self.log(f"{'='*60}\n")

            # Success dialog
            msg = f"{model_display} EoS fitting completed!"
            details = f"V₀ = {params.V0:.4f} ± {params.V0_err:.4f} Å³/atom\n"
            details += f"B₀ = {params.B0:.2f} ± {params.B0_err:.2f} GPa\n"
            details += f"B₀' = {params.B0_prime:.3f} ± {params.B0_prime_err:.3f}\n"
            if abs(params.B0_prime2) > 1e-10:
                details += f"B₀'' = {params.B0_prime2:.4f} GPa⁻¹\n"
            details += f"R² = {params.R_squared:.6f}"

            self.show_success_dialog("EoS Fitting Complete", msg, details)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = str(e)
            self.log(f"❌ Error during EoS fitting: {error_msg}")
            self.log(f"\nDetails:\n{error_details}")
            self.show_error("Error", f"EoS fitting failed:\n\n{error_msg}")
        finally:
            self.root.after(0, self.progress.stop)