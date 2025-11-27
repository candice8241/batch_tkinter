import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter
import os
import pandas as pd
from scipy.special import wofz

# ---------- Voigt ----------
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def fit_voigt(xdata, ydata, p0=None):
    if p0 is None:
        amplitude_guess = np.max(ydata)
        center_guess = xdata[np.argmax(ydata)]
        sigma_guess = np.std(xdata) / 5
        gamma_guess = sigma_guess
        p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]
    bounds = ([0, xdata.min(), 0, 0], [np.inf, xdata.max(), np.inf, np.inf])
    popt, pcov = curve_fit(voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    return popt, pcov

# ---------- Pseudo-Voigt ----------
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def fit_pseudo_voigt(xdata, ydata, p0=None):
    if p0 is None:
        amplitude_guess = np.max(ydata)
        center_guess = xdata[np.argmax(ydata)]
        sigma_guess = np.std(xdata) / 5
        gamma_guess = sigma_guess
        eta_guess = 0.5
        p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess, eta_guess]
    bounds = ([0, xdata.min(), 0, 0, 0], [np.inf, xdata.max(), np.inf, np.inf, 1.0])
    popt, pcov = curve_fit(pseudo_voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    return popt, pcov

# ---------- Batch Fitter Class ----------
class BatchFitter:
    def __init__(self, folder, fit_method="pseudo"):
        self.folder = folder
        self.save_dir = os.path.join(folder, "fit_output")
        os.makedirs(self.save_dir, exist_ok=True)
        self.fit_method = fit_method.lower()

    def process_file(self, file_path):
        try:
            with open(file_path, encoding='latin1') as f:
                data = np.genfromtxt(f, comments="#")
            x = data[:, 0]
            y = data[:, 1]
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")
            return

        filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n📄 Processing file: {filename}")

        # Find all peaks with width info
        all_peaks, _ = find_peaks(y, distance=20)
        results_half = peak_widths(y, all_peaks, rel_height=0.5)  
        peak_width_dict = {p: w for p, w in zip(all_peaks, results_half[0])}

        # ➤ Smooth the signal to calculate derivatives safely
        y_smooth = savgol_filter(y, window_length=11, polyorder=2)
        
        filtered_peaks = []
        window = 80
        
        for i in all_peaks:
            if i <= window or i >= len(y) - window:
                continue
        
            center = y[i]
            # ➤ Calculate mean intensity in 10% left and right regions        
            left_region = y[i - window : i - window + int(window * 0.1)]
            right_region = y[i + window - int(window * 0.1) : i + window]
            left_mean = np.mean(left_region)
            right_mean = np.mean(right_region)
        
            # ➤ Peak must be higher than both left and right average (by 15%)
            if center < left_mean * 1.15 or center < right_mean * 1.15:
                continue
        
            # ➤ Peak center should not be a valley (exclude sharp dips)
            if center < y[i - 10] and center < y[i + 10]:
                continue
        
            # ➤ Use second derivative: should be negative (indicates a peak)
            second_derivative = y_smooth[i + 1] - 2 * y_smooth[i] + y_smooth[i - 1]
            if second_derivative > 0:
                continue
        
            filtered_peaks.append(i)

        if not filtered_peaks:
            print("⚠️ No valid peaks found, skipping this file.")
            return
        
        n_peaks = len(filtered_peaks)
        ncols = 3
        nrows = int(np.ceil(n_peaks / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axs = axs.flatten()
        
        results = []

    
        for idx, peak in enumerate(filtered_peaks):
            # ---- Adaptive window based on peak width ----
            width_pts = peak_width_dict.get(peak, 40)  
            window = int(width_pts * 1.1)              
            window = max(20, min(window, 100))         
        
            # ---- Extract local region ----
            left = max(0, peak - window)
            right = min(len(x), peak + window)
            x_local = x[left:right]
            y_local = y[left:right]
        
            # ---- Background estimation ----
            split_index = max(1, int(len(y_local) * 0.05))
            
            # Left background (lowest 10% average)
            left_y = y_local[:split_index]
            left_x = x_local[:split_index]
            N_left = max(1, int(len(left_y) * 0.1))
            low_left_idx = np.argsort(left_y)[:N_left]
            bg_left_y = np.mean(left_y[low_left_idx])
            bg_left_x = np.mean(left_x[low_left_idx])
            
            # Right background (lowest 10% average)
            right_y = y_local[-split_index:]
            right_x = x_local[-split_index:]
            N_right = max(1, int(len(right_y) * 0.1))
            low_right_idx = np.argsort(right_y)[:N_right]
            bg_right_y = np.mean(right_y[low_right_idx])
            bg_right_x = np.mean(right_x[low_right_idx])
            
            # ---- Background line and subtraction ----
            slope = (bg_right_y - bg_left_y) / (bg_right_x - bg_left_x)
            background = bg_left_y + slope * (x_local - bg_left_x)
            y_fit_input = y_local - background

            x_smooth = np.linspace(x_local.min(), x_local.max(), 5000)

            try:
                if self.fit_method == "voigt":
                    popt, _ = fit_voigt(x_local, y_fit_input)
                    y_fit = voigt(x_smooth, *popt)
                    results.append({
                        "Peak #": idx+1, "Center": popt[1], "Amplitude": popt[0],
                        "Sigma": popt[2], "Gamma": popt[3], "Eta": "N/A"
                    })
                else:
                    popt, _ = fit_pseudo_voigt(x_local, y_fit_input)
                    y_fit = pseudo_voigt(x_smooth, *popt)
                    results.append({
                        "Peak #": idx+1, "Center": popt[1], "Amplitude": popt[0],
                        "Sigma": popt[2], "Gamma": popt[3], "Eta": popt[4]
                    })

                bg_line = bg_left_y + slope * (x_smooth - bg_left_x)

                full_fit = y_fit + bg_line

                ax = axs[idx]
                ax.plot(x_local, y_local, color='black', label="Raw Data")
                ax.plot(x_smooth, full_fit, color='#BA55D3', linestyle='--', linewidth=2, label=f"{self.fit_method.capitalize()} Fit")
                ax.plot(x_smooth, bg_line, color='#FF69B4', linestyle='-', linewidth=1.5, label="Background")
                ax.set_title(f"Peak {idx+1} @ {popt[1]:.3f}")
                ax.set_xlabel("2θ (degree)")
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.set_facecolor("white")
                ax.grid(False)

            except RuntimeError:
                print(f"⚠️ Peak {idx+1} fit failed 💔")
                axs[idx].text(0.3, 0.5, "Fit failed", transform=axs[idx].transAxes, fontsize=12, color='red')

        for j in range(len(filtered_peaks), len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"{filename} - {self.fit_method.capitalize()} Fit", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig_path = os.path.join(self.save_dir, f"{filename}_fit.png")
        plt.savefig(fig_path)
        plt.close()

        df = pd.DataFrame(results)
        df["File"] = filename
        df.to_csv(os.path.join(self.save_dir, f"{filename}_results.csv"), index=False)
        return df

    def run_batch_fitting(self):
        files = sorted(f for f in os.listdir(self.folder) if f.endswith(".xy"))
        all_dfs = []

        for fname in files:
            fpath = os.path.join(self.folder, fname)
            df = self.process_file(fpath)
            if df is not None:
                all_dfs.append(df)
                all_dfs.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))  # add blank row

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_csv_path = os.path.join(self.save_dir, "all_results.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"\n📦 Total results saved to: {combined_csv_path}")

# ---------- Execution ----------
if __name__ == "__main__":
    folder_path = r"D:\HEPS\ID31\dioptas_data\test"  # update if needed
    fitter = BatchFitter(folder=folder_path, fit_method="pseudo")  # or "voigt"
    fitter.run_batch_fitting()
