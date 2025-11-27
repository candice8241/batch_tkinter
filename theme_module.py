# -*- coding: utf-8 -*-
"""
GUI Components Module for Curve Fitting Application
Contains all UI elements, components, color schemes, and utility methods

Created on Fri Nov 14 09:31:22 2025
@author: 16961
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import math
from pathlib import Path


# ==============================================================================
# Modern UI Components
# ==============================================================================

class ModernButton(tk.Canvas):
    """Modern button component with rounded corners and hover effects"""

    def __init__(self, parent, text, command, icon="", bg_color="#9D4EDD",
                 hover_color="#C77DFF", text_color="white", width=200, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                        highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color

        self.rect = self.create_rounded_rectangle(0, 0, width, height, radius=10,
                                                   fill=bg_color, outline="")

        display_text = f"{icon}  {text}" if icon else text
        self.text_id = self.create_text(width//2, height//2, text=display_text,
                                       fill=text_color, font=('Comic Sans MS', 11, 'bold'))

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.config(cursor="hand2")

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        """Create a rounded rectangle using polygon with smooth curves"""
        points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1,
                  x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2,
                  x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2, x1, y2,
                  x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, e):
        """Handle mouse enter event - change to hover color"""
        self.itemconfig(self.rect, fill=self.hover_color)

    def on_leave(self, e):
        """Handle mouse leave event - restore normal color"""
        self.itemconfig(self.rect, fill=self.bg_color)

    def on_click(self, e):
        """Handle button click event"""
        if self.command:
            self.command()


class ModernTab(tk.Frame):
    """Modern tab component with active/inactive states"""

    def __init__(self, parent, text, command, is_active=False, **kwargs):
        super().__init__(parent, bg=parent['bg'], **kwargs)
        self.command = command
        self.is_active = is_active
        self.parent_widget = parent

        self.active_color = "#9D4EDD"
        self.inactive_color = "#8B8BA7"
        self.hover_color = "#C77DFF"

        self.label = tk.Label(self, text=text,
                             fg=self.active_color if is_active else self.inactive_color,
                             bg=parent['bg'], font=('Comic Sans MS', 11, 'bold'),
                             cursor="hand2", padx=20, pady=10)
        self.label.pack()

        self.underline = tk.Frame(self, bg=self.active_color if is_active else parent['bg'],
                                 height=3)
        self.underline.pack(fill=tk.X)

        self.label.bind("<Enter>", self.on_enter)
        self.label.bind("<Leave>", self.on_leave)
        self.label.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        """Handle mouse enter event"""
        if not self.is_active:
            self.label.config(fg=self.hover_color)

    def on_leave(self, e):
        """Handle mouse leave event"""
        if not self.is_active:
            self.label.config(fg=self.inactive_color)

    def on_click(self, e):
        """Handle tab click event"""
        if self.command:
            self.command()

    def set_active(self, active):
        """Set the active state of the tab"""
        self.is_active = active
        self.label.config(fg=self.active_color if active else self.inactive_color)
        self.underline.config(bg=self.active_color if active else self.parent_widget['bg'])


class CuteSheepProgressBar(tk.Canvas):
    """Cute sheep progress bar animation"""

    def __init__(self, parent, width=700, height=80, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                        highlightthickness=0, **kwargs)

        self.width = width
        self.height = height
        self.sheep = []
        self.is_animating = False
        self.frame_count = 0

    def draw_adorable_sheep(self, x, y, jump_phase):
        """Draw a cute sheep emoji with bounce animation"""
        jump = -abs(math.sin(jump_phase) * 15)
        y_pos = y + jump

        # Draw large animated sheep emoji 🐑
        # Use text to draw emoji with larger font size
        self.create_text(x, y_pos, text="🐿️", font=("Segoe UI Emoji", 48), anchor="center")

    def start(self):
        """Start the animation"""
        self.is_animating = True
        self.frame_count = 0
        self.sheep = []
        self._animate()

    def stop(self):
        """Stop the animation and clear the canvas"""
        self.is_animating = False
        self.delete("all")
        self.sheep = []
        self.frame_count = 0

    def _animate(self):
        """Internal animation loop"""
        if not self.is_animating:
            return

        self.delete("all")

        # Spawn new sheep periodically
        if self.frame_count % 35 == 0:
            self.sheep.append({'x': -40, 'phase': 0})

        # Update and draw all sheep
        new_sheep = []
        for sheep_data in self.sheep:
            sheep_data['x'] += 3.5
            sheep_data['phase'] += 0.25

            if sheep_data['x'] < self.width + 50:
                self.draw_adorable_sheep(sheep_data['x'], self.height // 2, sheep_data['phase'])
                new_sheep.append(sheep_data)

        self.sheep = new_sheep
        self.frame_count += 1

        self.after(35, self._animate)


# ==============================================================================
# Base GUI Class with Utilities
# ==============================================================================

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
        """Create a styled card frame with border"""
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
        """Open file browser dialog and set the selected file path"""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            variable.set(filename)

    def browse_pattern(self, variable, filetypes):
        """Open file browser and create pattern from selected file"""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            folder = os.path.dirname(filename)
            ext = os.path.splitext(filename)[1]
            pattern = os.path.join(folder, f"*{ext}")
            variable.set(pattern)

    def browse_folder(self, variable):
        """Open folder browser dialog and set the selected folder path"""
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def show_success(self, root, message):
        """Show success dialog with cute styling"""
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