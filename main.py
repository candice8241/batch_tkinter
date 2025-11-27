# -*- coding: utf-8 -*-
"""
Main GUI Application
XRD Data Post-Processing Suite - Entry point and main window
"""

import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
import os
from PIL import Image, ImageTk
import sys
import ctypes
from theme_module import GUIBase, ModernButton, ModernTab, CuteSheepProgressBar
from powder_module import PowderXRDModule
from radial_module import AzimuthalIntegrationModule
from single_crystal_module import SingleCrystalModule


class XRDProcessingGUI(GUIBase):
    """Main GUI application for XRD data processing"""

    def __init__(self, root):
        """
        Initialize main GUI

        Args:
            root: Tk root window
        """
        super().__init__()
        self.root = root
        self.root.title("XRD Data Post-Processing")
        self.root.geometry("1100x950")
        self.root.resizable(True, True)

        # Try to set icon (Windows only)
        try:
            icon_path = r'D:\HEPS\ID31\dioptas_data\github_felicity\batch\\HP_full_package\ChatGPT Image.ico'
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not load icon: {e}")

        self.root.configure(bg=self.colors['bg'])

        # Initialize modules
        self.powder_module = None
        self.radial_module = None
        self.single_crystal_module = None

        # Containers for each module (packed/unpacked instead of destroyed)
        self.module_frames = {
            "powder": None,
            "single": None,
            "radial": None
        }

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup main user interface"""
        # Header section
        header_frame = tk.Frame(self.root, bg=self.colors['card_bg'], height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        header_content = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(header_content, text="", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 32)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header_content, text="XRD Data Post_Processing",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Arial', 20, 'bold')).pack(side=tk.LEFT)

        # Tab bar
        tab_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        tab_frame.pack(fill=tk.X, padx=30, pady=(15, 0))

        tabs_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        tabs_container.pack(side=tk.LEFT)

        self.powder_tab = ModernTab(tabs_container, "Powder XRD",
                                    lambda: self.switch_tab("powder"), is_active=True)
        self.powder_tab.pack(side=tk.LEFT, padx=(0, 15))

        self.single_tab = ModernTab(tabs_container, "Single Crystal XRD",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=15)

        self.radial_tab = ModernTab(tabs_container, "Radial XRD",
                                   lambda: self.switch_tab("radial"))
        self.radial_tab.pack(side=tk.LEFT, padx=15)

        # Scrollable container setup
        container = tk.Frame(self.root, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        canvas = tk.Canvas(container, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        self.scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.root.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas

        # Prebuild module UIs so the first visible load is already prepared
        self.prebuild_modules()

        # Show powder tab by default
        self.switch_tab("powder")

    def _ensure_frame(self, name):
        if self.module_frames[name] is None:
            self.module_frames[name] = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        return self.module_frames[name]

    def prebuild_modules(self):
        """Construct all module frames and their UIs ahead of first use to avoid initial flash."""
        powder_frame = self._ensure_frame("powder")
        if self.powder_module is None:
            self.powder_module = PowderXRDModule(powder_frame, self.root)
            self.powder_module.setup_ui()

        radial_frame = self._ensure_frame("radial")
        if self.radial_module is None:
            self.radial_module = AzimuthalIntegrationModule(radial_frame, self.root)
            self.radial_module.setup_ui()

        single_frame = self._ensure_frame("single")
        if self.single_crystal_module is None:
            self.single_crystal_module = SingleCrystalModule(single_frame, self.root)
            self.single_crystal_module.setup_ui()

    def switch_tab(self, tab_name):
        """
        Switch between main tabs

        Args:
            tab_name: Name of tab to switch to ('powder', 'single', 'radial')
        """
        # Update tab active states
        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        # Hide all module frames instead of destroying
        for frame in self.module_frames.values():
            if frame is not None:
                frame.pack_forget()

        target_frame = None

        # Load appropriate module (create once, then just re-pack to avoid flicker)
        if tab_name == "powder":
            target_frame = self._ensure_frame("powder")
            if self.powder_module is None:
                self.powder_module = PowderXRDModule(target_frame, self.root)
                self.powder_module.setup_ui()

        elif tab_name == "radial":
            target_frame = self._ensure_frame("radial")
            if self.radial_module is None:
                self.radial_module = AzimuthalIntegrationModule(target_frame, self.root)
                self.radial_module.setup_ui()

        elif tab_name == "single":
            target_frame = self._ensure_frame("single")
            if self.single_crystal_module is None:
                self.single_crystal_module = SingleCrystalModule(target_frame, self.root)
                self.single_crystal_module.setup_ui()

        if target_frame is not None:
            target_frame.pack(fill=tk.BOTH, expand=True)
            self.root.update_idletasks()


def launch_main_app():
    # Set AppUserModelID (important for taskbar icon)
    app_id = u"mycompany.myapp.xrdpostprocessor"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    # Create main window
    root = tk.Tk()

    #Set window icon (taskbar + title bar)
    icon_path = r"D:\HEPS\ID31\dioptas_data\github_felicity\batch\\HP_full_package\ChatGPT Image.ico"
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Failed to set icon: {e}")
    else:
        print("Icon file not found!")

    #Set window title
    root.title("XRD Data Post-Processing")

    #Set window size and allow resizing
    root.geometry("700x400")
    root.resizable(True, True)  # width & height resizable

    #Set background color to purple-pink
    root.configure(bg="#EDE9F3")  # light purple-pink (Thistle)

    #Define cute font style
    cute_font = tkFont.Font(family="Arial", size=14, weight="bold")

    # Create adorable welcome label
    welcome_text = (" Hey there！ Ready to sparkle your XRD data? \n"
                    "\n"
                    #"💜 Beam me up, XRD Commander – it's fitting time! ~ 💖✨\n"
                    "\n"
                    "📧 Contact: lixd@ihep.ac.cn\n\n fzhang@ihep.ac.cn\n\n yswang@ihep.ac.cn")
    label = tk.Label(
        root,
        text=welcome_text,
        font=cute_font,
        bg="#F9EBF2",
        fg="#0A0A0A",
        pady=90
    )
    label.pack(pady=40)

    root.mainloop()

def show_startup_window():
    splash = tk.Tk()
    splash.title("Loading...")
    
    # Set a custom application icon (Windows only)
    icon_path = r"D:\HEPS\ID31\dioptas_data\github_felicity\batch\HP_full_package\ChatGPT Image.ico"
    if os.path.exists(icon_path):
        splash.iconbitmap(icon_path)
    
    # Set Windows Taskbar App ID
    app_id = u"mycompany.myapp.xrdpostprocessor"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    
    # Window style
    window_width = 480
    window_height = 220
    
    # Get screen dimensions and center the window
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    splash.geometry(f"{window_width}x{window_height}+{x}+{y}")
    splash.configure(bg="#fbeaff")  # pastel pink
    splash.resizable(False, False)
    
    # Message
    title_label = tk.Label(
        splash,
        text="Starting up, please wait...",
        font=("Arial", 12, "bold"),
        fg="#ab47bc",  # orchid tone
        bg="#fbeaff"
    )
    title_label.pack(pady=(15, 5))
    
    # Progress percentage label
    percent_label = tk.Label(
        splash,
        text="0%",
        font=("Arial", 10),
        fg="#9c27b0",
        bg="#fbeaff"
    )
    percent_label.pack(pady=5)
    
    # Canvas for sheep progress bar
    canvas_width = 400
    canvas_height = 60
    canvas = tk.Canvas(splash, width=canvas_width, height=canvas_height, 
                       bg="#fbeaff", highlightthickness=0)
    canvas.pack(pady=15)
    
    # Draw progress bar background (track)
    track_height = 30
    track_y = (canvas_height - track_height) // 2
    canvas.create_rectangle(50, track_y, canvas_width - 50, track_y + track_height,
                           fill="#f3cfe2", outline="#d8b4e2", width=2)
    
    # Total sheep that will fill the progress bar (5 sheep = 100%)
    total_sheep = 5
    sheep_spacing = (canvas_width - 100) / total_sheep
    sheep_objects = []
    
    def animate_progress(progress=0):
        if progress <= 100:
            # Update percentage
            percent_label.config(text=f"{progress}%")
            
            # Calculate how many sheep should be visible (1 sheep per 20%)
            sheep_to_show = progress // 20
            
            # Add sheep if needed
            while len(sheep_objects) < sheep_to_show:
                sheep_index = len(sheep_objects)
                x_pos = 50 + sheep_spacing * sheep_index + sheep_spacing / 2
                y_pos = track_y + track_height / 2
                
                # Create new sheep in the progress bar
                sheep = canvas.create_text(x_pos, y_pos, 
                                          text="🐑", 
                                          font=("Segoe UI Emoji", 20),
                                          anchor="center")
                sheep_objects.append(sheep)
                
                # Bounce animation for new sheep
                def bounce(obj, bounces=2, offset=-6):
                    if bounces > 0:
                        canvas.move(obj, 0, offset)
                        splash.after(80, lambda o=obj, b=bounces, off=offset: 
                                   bounce(o, b - 1, -off))
                
                bounce(sheep)
            
            # Update title with fun messages at milestones
            if progress == 20:
                title_label.config(text="Loading modules... 🌸")
            elif progress == 40:
                title_label.config(text="Setting up workspace... 💜")
            elif progress == 60:
                title_label.config(text="Almost there, sweetheart! 💖")
            elif progress == 80:
                title_label.config(text="Final touches... ✨")
            elif progress == 100:
                title_label.config(text="Ready to go! 🎉")
            
            # Continue animation
            delay = 35 if progress < 90 else 25
            splash.after(delay, lambda: animate_progress(progress + 2))
        else:
            # Finish animation with short delay
            splash.after(100, lambda: [splash.destroy(), launch_main_app()])
    
    animate_progress()
    splash.mainloop()
    
if __name__ == "__main__":
    show_startup_window()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = XRDProcessingGUI(root)  # Construct your main GUI here
    root.mainloop()

if __name__ == "__main__":
    
    main()
    
    