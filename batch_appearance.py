# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 09:31:22 2025

@author: 16961
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from batch_integration import BatchIntegrator
from peak_fitting import BatchFitter
from birch_murnaghan_batch import BirchMurnaghanFitter
from batch_cal_volume import XRayDiffractionAnalyzer
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
import re
import warnings
import math
import random



class ModernButton(tk.Canvas):
    """Modern button component"""
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
        points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1,
                  x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2,
                  x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2, x1, y2,
                  x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, e):
        self.itemconfig(self.rect, fill=self.hover_color)

    def on_leave(self, e):
        self.itemconfig(self.rect, fill=self.bg_color)

    def on_click(self, e):
        if self.command:
            self.command()


class ModernTab(tk.Frame):
    """Modern tab component"""
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
        if not self.is_active:
            self.label.config(fg=self.hover_color)

    def on_leave(self, e):
        if not self.is_active:
            self.label.config(fg=self.inactive_color)

    def on_click(self, e):
        if self.command:
            self.command()

    def set_active(self, active):
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
        jump = -abs(math.sin(jump_phase) * 20)
        y = y + jump

        # Shadow
        self.create_oval(x-15, y+25, x+15, y+28, fill="#E8E4F3", outline="")
        # Body
        self.create_oval(x-20, y-15, x+20, y+15, fill="#FFFFFF", outline="#FFB6D9", width=3)
        self.create_oval(x-18, y-10, x-10, y-2, fill="#FFF5FF", outline="")
        self.create_oval(x+10, y-8, x+18, y, fill="#FFF5FF", outline="")
        self.create_oval(x-5, y+8, x+5, y+15, fill="#FFF5FF", outline="")
        # Head
        self.create_oval(x+15, y-12, x+35, y+8, fill="#FFE4F0", outline="#FFB6D9", width=3)
        # Ears
        self.create_polygon(x+17, y-10, x+20, y-18, x+23, y-10,
                           fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)
        self.create_polygon(x+27, y-10, x+30, y-18, x+33, y-10,
                           fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)
        # Eyes
        self.create_oval(x+19, y-6, x+24, y-1, fill="#FFFFFF")
        self.create_oval(x+20, y-5, x+23, y-2, fill="#2B2D42")
        self.create_oval(x+21, y-4, x+22, y-3, fill="#FFFFFF")
        self.create_oval(x+26, y-6, x+31, y-1, fill="#FFFFFF")
        self.create_oval(x+27, y-5, x+30, y-2, fill="#2B2D42")
        self.create_oval(x+28, y-4, x+29, y-3, fill="#FFFFFF")
        # Nose and mouth
        self.create_oval(x+23, y+2, x+27, y+6, fill="#FFB6D9", outline="#FF6B9D", width=2)
        self.create_arc(x+20, y+3, x+30, y+9, start=0, extent=-180,
                       outline="#FF6B9D", width=3, style="arc")
        # Cheeks
        self.create_oval(x+16, y+1, x+19, y+4, fill="#FFD4E5", outline="")
        self.create_oval(x+31, y+1, x+34, y+4, fill="#FFD4E5", outline="")

        # Legs with animation
        leg_offset = abs(math.sin(jump_phase) * 3)
        self.create_line(x-12, y+15, x-12, y+24-leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        self.create_line(x-4, y+15, x-4, y+24+leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        self.create_line(x+6, y+15, x+6, y+24-leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        self.create_line(x+14, y+15, x+14, y+24+leg_offset, fill="#FFB6D9", width=5, capstyle="round")

        # Hooves
        self.create_oval(x-14, y+22-leg_offset, x-10, y+25-leg_offset, fill="#D4BBFF")
        self.create_oval(x-6, y+22+leg_offset, x-2, y+25+leg_offset, fill="#D4BBFF")
        self.create_oval(x+4, y+22-leg_offset, x+8, y+25-leg_offset, fill="#D4BBFF")
        self.create_oval(x+12, y+22+leg_offset, x+16, y+25+leg_offset, fill="#D4BBFF")

        # Tail
        self.create_oval(x-22, y+5, x-16, y+11, fill="#FFFFFF", outline="#FFB6D9", width=2)

    def start(self):
        self.is_animating = True
        self.frame_count = 0
        self.sheep = []
        self._animate()

    def stop(self):
        self.is_animating = False
        self.delete("all")
        self.sheep = []
        self.frame_count = 0

    def _animate(self):
        if not self.is_animating:
            return

        self.delete("all")

        # Spawn new sheep periodically
        if self.frame_count % 35 == 0:
            self.sheep.append({'x': -40, 'phase': 0})

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