# -*- coding: utf-8 -*-
"""
Base GUI Components and Styles
Contains shared UI elements, color schemes, and utility methods
"""

import tkinter as tk
from tkinter import filedialog
from batch_appearance import ModernButton


class GUIBase:
    """Base class for GUI components with shared styles and utilities"""

    def __init__(self):
        """Initialize color scheme and styles"""
        self.colors = {
            'bg': '#F8F7FF',
            'card_bg': '#FFFFFF',
            'primary': '#B794F6',
            'primary_hover': '#D4BBFF',
            'secondary': '#E0AAFF',
            'accent': '#FF6B9D',
            'text_dark': '#2B2D42',
            'text_light': '#8B8BA7',
            'border': '#E8E4F3',
            'success': '#06D6A0',
            'error': '#EF476F',
            'light_purple': '#E6D9F5',
            'active_module': '#C8B3E6'
        }

    def create_card_frame(self, parent, bg=None, **kwargs):
        """Create a styled card frame"""
        if bg is None:
            bg = self.colors['card_bg']
        card = tk.Frame(parent, bg=bg, relief='flat', borderwidth=0, **kwargs)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        return card

    def create_file_picker(self, parent, label, variable, filetypes, pattern=False):
        """Create a file picker widget with browse button"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        if pattern:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_pattern(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        else:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_file(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_folder_picker(self, parent, label, variable):
        """Create a folder picker widget with browse button"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        btn = ModernButton(input_frame, "Browse",
                         lambda: self.browse_folder(variable),
                         bg_color=self.colors['secondary'],
                         hover_color=self.colors['primary'],
                         width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_entry(self, parent, label, variable):
        """Create a text entry widget"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        entry = tk.Entry(container, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1)
        entry.pack(fill=tk.X, ipady=3)

    def browse_file(self, variable, filetypes):
        """Open file browser dialog"""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            variable.set(filename)

    def browse_pattern(self, variable, filetypes):
        """Open file browser and create pattern from selected file"""
        import os
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            folder = os.path.dirname(filename)
            ext = os.path.splitext(filename)[1]
            pattern = os.path.join(folder, f"*{ext}")
            variable.set(pattern)

    def browse_folder(self, variable):
        """Open folder browser dialog"""
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def show_success(self, root, message):
        """Show success dialog"""
        dialog = tk.Toplevel(root)
        dialog.title("Success")
        dialog.geometry("450x300")
        dialog.configure(bg=self.colors['card_bg'])
        dialog.resizable(False, False)
        dialog.transient(root)
        dialog.grab_set()

        tk.Label(dialog, text="✅", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 64)).pack(pady=(30, 20))

        tk.Label(dialog, text=message, bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Comic Sans MS', 13, 'bold'),
                wraplength=400).pack(pady=(10, 30))

        ModernButton(dialog, "OK", dialog.destroy,
                    bg_color=self.colors['primary'],
                    hover_color=self.colors['primary_hover'],
                    width=120, height=40).pack()