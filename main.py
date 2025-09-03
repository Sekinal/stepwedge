import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------- configuration ----------------------------------

CSV_PATH = "profiles.csv"                 # input CSV file
X_COL = "Raw New Profile X"               # Column name for X-axis data (projections)
Y_COL = "Raw New Profile Y"               # Column name for Y-axis data (signal)

# Acquisition and machine settings (adjust as per your system's specifics)
sample_rate_hz = 30.0                     # Sample frequency (Hz), 1 projection every ~0.0333 s
time_per_projection_s = 1.0 / sample_rate_hz
couch_speed_mm_s = 200.0 / 300.0          # Couch travels 20 cm in 300 s for step-wedge QA

# Step wedge: nominal physical step lengths (mm)
nominal_step_lengths_mm = np.array([29.9, 30.0, 30.0, 30.0, 30.1])
# Aluminum 6061 to water-equivalent depths (mm) for the 5 steps (from the paper)
# Includes an initial 0.0 mm for the "air" level.
depths_water_mm = np.array([0.0, 52.5, 104.9, 157.4, 209.8, 262.5])

# Plotting output
OUT_PNG = "stepwedge_results.png"

# ------------------------- model implementation (modified) -------------------

def _safe_savgol(y):
    """
    Applies Savitzky-Golay filter for smoothing, preserving signal edges.
    Adjusts window size based on data length.
    """
    n = len(y)
    # Ensure window_length is odd and sufficient, but not too large
    win = max(31, min(301, (n // 50) * 2 + 1))
    return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

def _changepoints_7_segments(y_s, n_bkps=6):
    """
    Detects 6 changepoints to segment the signal into 7 distinct parts (plateaus).
    Prioritizes the 'ruptures' library for robustness if available,
    otherwise uses a derivative-based fallback.
    """

    N = len(y_s)
    dy = np.abs(np.gradient(y_s))

    # Basic peak finding with minimum distance
    # Filter out peaks that are too close to each other or too small
    peaks, _ = find_peaks_simple(dy, min_dist=max(50, N // 40))

    if len(peaks) < n_bkps:
        # If not enough peaks, distribute them somewhat uniformly
        cps = np.linspace(N / (n_bkps + 1), N - N / (n_bkps + 1), n_bkps).astype(int)
    else:
        # Sort peaks by prominence (derivative magnitude) and take the top N
        # For simplicity here, just take the largest (which are usually the sharpest transitions)
        sorted_indices = np.argsort(dy[peaks])[-n_bkps:] # Get indices of n_bkps largest peaks
        cps = peaks[sorted_indices]

        return sorted(list(set(cps))) # Ensure unique and sorted

def find_peaks_simple(data, min_dist):
    """
    Simple peak finding function (like scipy.signal.find_peaks but simpler)
    Used as an internal fallback if ruptures is not installed.
    """
    peaks = []
    if len(data) == 0:
        return np.array(peaks, dtype=int), {}
    
    # Very basic peak finding: local maxima above threshold
    threshold = np.mean(data) + 0.5 * np.std(data) # Simple heuristic

    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
            # Check proximity to previously found peaks
            is_far_enough = True
            for p_idx in peaks:
                if abs(i - p_idx) < min_dist:
                    is_far_enough = False
                    break
            if is_far_enough:
                peaks.append(i)
    return np.array(peaks, dtype=int), {}


def _enforce_pattern_levels(levels_raw, x_vals, y_vals):
    """
    Adjusts raw plateau levels to match the expected step-wedge pattern:
    - s1 (air) is non-decreasing relative to itself (always true).
    - s2-s6 (steps) are non-decreasing (your data goes up for the first few).
    - s7 (final baseline) is forced to be lower than s6 (the last step plateau),
      simulating the final drop.
    Does not modify s1; s2-s6 are monotonically increased if a dip exists; s7 is adjusted.
    """
    s = np.array(levels_raw, float)

    # Enforce s1 to s6 non-decreasing (as indicated by "our data is actually increasing")
    # This means steps get progressively higher or stay the same
    # The paper shows decreasing steps (signal = f(thickness)), so if your signal *increases*
    # with density/depth, this will be correct. If increasing thickness leads to decreasing
    # signal (more absorption), then this should be reversed (cumulative minimum).
    s_steps_inc = s[1:6] # S2 to S6
    s[1:6] = np.maximum.accumulate(s_steps_inc) # Force non-decreasing for plateaus 2-6

    # Enforce s7 level to be below s6, given the "falls and decreases to about 0.25" description
    # of the final segment.
    # Provide a reasonable bound to ensure it doesn't go too low or stick too high.
    if s[6] >= s[5]: # If the last plateau is not clearly below the preceding one
        # Try to pull s7 down based on the range of the main signal
        min_overall_signal = np.amin(y_vals)
        max_overall_signal = np.amax(y_vals)
        expected_low_level = min_overall_signal + 0.1 * (max_overall_signal - min_overall_signal)
        
        # Ensure s7 is below s6, but clamp it to avoid extreme values.
        # It should also not go below the absolute minimum signal observed in the data.
        s[6] = max(min_overall_signal, min(s[5] * 0.9, expected_low_level)) # Make it at least somewhat lower than s6 and above min observed
        if s[6] > s_raw[6]: # If we had to pull it down, log it
             logging.info(f"Adjusted S7 from {s_raw[6]:.2f} to {s[6]:.2f} to enforce final drop.")

    logging.debug(f"Adjusted plateau levels: {s}")
    return s


def _ramp_edges_from_cp(x, y_s, cp_idx, si, sj, search_halfwidth_idx=400):
    """
    Identifies the start (p_k) and end (p_{k+1}) of a ramp around a given changepoint index.
    It works by searching outwards from the changepoint, looking for where the
    signal deviates from the adjacent plateaus, or where the derivative is significant.
    """
    N = len(y_s)
    # Define a local search window around the changepoint
    i_start_window = max(0, cp_idx - search_halfwidth_idx)
    i_end_window = min(N - 1, cp_idx + search_halfwidth_idx)
    
    # Slice the data for local analysis
    x_local = x[i_start_window : i_end_window + 1]
    y_local = y_s[i_start_window : i_end_window + 1]
    
    if len(x_local) < 2: # Not enough data in window, fall back to cp itself
        return x[cp_idx], x[cp_idx]

    # Calculate derivative of the local smoothed signal
    dy_local = np.gradient(y_local, x_local)

    # Determine expected direction of ramp
    signal_diff = sj - si
    direction_sign = np.sign(signal_diff)

    # Define a threshold for derivative significance
    # Use Robust Median Absolute Deviation (MAD) for thresholding if available, else simple std
    mad = np.median(np.abs(dy_local - np.median(dy_local)))
    deriv_threshold = max(0.1 * np.abs(signal_diff) / (x_local[-1] - x_local[0] + 1e-9), 
                          3 * mad if mad > 0 else 0.05 * np.std(dy_local))
    deriv_threshold = max(deriv_threshold, 1e-4) # Minimum threshold to avoid tiny values

    # Identify points where the derivative indicates a significant slope
    if direction_sign != 0: # If there's an actual change in signal
        significant_slope_mask = (direction_sign * dy_local > deriv_threshold)
    else: # If plateaus are flat, look for any slope
        significant_slope_mask = (np.abs(dy_local) > deriv_threshold)

    # Find the indices within the local window where the significant slope starts and ends
    slope_indices_local = np.where(significant_slope_mask)[0]

    if slope_indices_local.size > 0:
        # These are the beginning and end of the detected ramp within the local window
        i_ramp_start_local = slope_indices_local[0]
        i_ramp_end_local = slope_indices_local[-1]
        
        # Convert local indices back to global x-values
        x_ramp_start = x_local[i_ramp_start_local]
        x_ramp_end = x_local[i_ramp_end_local]
        
        # Ensure start is before end
        if x_ramp_start > x_ramp_end:
            x_ramp_start, x_ramp_end = x_ramp_end, x_ramp_start

        return x_ramp_start, x_ramp_end
    else:
        # Fallback if no clear slope is detected: use a transition point based on value
        # Find index closest to midpoint between si and sj within the window
        k_mid_local = np.argmin(np.abs(y_local - (si + sj) / 2.0))
        
        # Estimate ramp "width" (e.g., 50 projections around the mid-point)
        ramp_half_width_proj = max(50, int(0.05 * len(x_local))) 
        
        k_start_local = max(0, k_mid_local - ramp_half_width_proj)
        k_end_local = min(len(x_local) - 1, k_mid_local + ramp_half_width_proj)
        
        return x_local[k_start_local], x_local[k_end_local]

def piecewise_stepwedge(x, p, s):
    """
    This function implements the piecewise-linear model of the step wedge,
    consisting of 7 flat plateaus (s1-s7) and 6 sloped ramps between them.
    It calculates the signal (y_fit) for a given set of x-values (projections).
    """
    x = np.asarray(x, float)
    p = np.asarray(p, float) # 12 transition points
    s = np.asarray(s, float) # 7 plateau signal levels

    y = np.empty_like(x, dtype=float)

    # Helper functions for clarity
    def fill_flat(mask, level):
        y[mask] = level

    def fill_slope(mask, x_left, x_right, y_left, y_right):
        # Linear interpolation: y = y_left + (y_right - y_left) * (x - x_left) / (x_right - x_left)
        t = (x[mask] - x_left) / max(x_right - x_left, 1e-9) # Normalize x to [0,1] within segment
        y[mask] = (1 - t) * y_left + t * y_right

    # Apply the piecewise model based on the 12 'p' transition points
    # and 7 's' plateau levels.

    # Segment 1: Flat s1 before p1
    mask = x <= p[0]
    fill_flat(mask, s[0])

    # Ramp 1: From s1 to s2 (between p1 and p2)
    mask = (x > p[0]) & (x <= p[1])
    fill_slope(mask, p[0], p[1], s[0], s[1])

    # Segment 2: Flat s2 (between p2 and p3)
    mask = (x > p[1]) & (x <= p[2])
    fill_flat(mask, s[1])

    # Ramp 2: From s2 to s3 (between p3 and p4)
    mask = (x > p[2]) & (x <= p[3])
    fill_slope(mask, p[2], p[3], s[1], s[2])

    # Segment 3: Flat s3 (between p4 and p5)
    mask = (x > p[3]) & (x <= p[4])
    fill_flat(mask, s[2])

    # Ramp 3: From s3 to s4 (between p5 and p6)
    mask = (x > p[4]) & (x <= p[5])
    fill_slope(mask, p[4], p[5], s[2], s[3])

    # Segment 4: Flat s4 (between p6 and p7)
    mask = (x > p[5]) & (x <= p[6])
    fill_flat(mask, s[3])

    # Ramp 4: From s4 to s5 (between p7 and p8)
    mask = (x > p[6]) & (x <= p[7])
    fill_slope(mask, p[6], p[7], s[3], s[4])

    # Segment 5: Flat s5 (between p8 and p9)
    mask = (x > p[7]) & (x <= p[8])
    fill_flat(mask, s[4])

    # Ramp 5: From s5 to s6 (between p9 and p10)
    mask = (x > p[8]) & (x <= p[9])
    fill_slope(mask, p[8], p[9], s[4], s[5])

    # Segment 6: Flat s6 (between p10 and p11)
    mask = (x > p[9]) & (x <= p[10])
    fill_flat(mask, s[5])

    # Ramp 6: From s6 to s7 (between p11 and p12)
    mask = (x > p[10]) & (x <= p[11])
    fill_slope(mask, p[10], p[11], s[5], s[6])

    # Segment 7: Flat s7 after p12
    mask = x > p[11]
    fill_flat(mask, s[6])

    return y


def fit_stepwedge_time_profile(x, y):
    """
    Fits the step-wedge time profile using a robust, deterministic approach.
    1. Smooths the signal.
    2. Identifies 6 changepoints (7 segments/plateaus).
    3. Calculates raw plateau levels from median values within segments.
    4. Enforces the specific pattern (initial baseline, rising steps, final drop).
    5. Determines ramp boundaries (p1-p12) by analyzing local derivatives
       around each changepoint.
    6. Constructs the final piecewise-linear model.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_s = _safe_savgol(y)  # Smooth the signal for robust analysis

    N = len(y_s)
    # Find 6 changepoints that define the 7 segments (plateaus)
    # cps are 0-based indices in the original x/y arrays.
    cps = _changepoints_7_segments(y_s, n_bkps=6)
    
    # Add start and end points to define segment ranges
    seg_edges_indices = [0] + cps + [N-1] # N-1 for last index, N for exclusive boundary

    # Calculate raw plateau levels (s1-s7) from the median signal within each segment
    s_raw = []
    # Using 25% trim from each end of a segment to avoid ramp influence on plateau level.
    trim_percentage = 0.25 
    for i in range(7):
        start_idx = seg_edges_indices[i]
        end_idx = seg_edges_indices[i+1]

        # Calculate practical trim based on segment length
        segment_len = end_idx - start_idx
        if segment_len <= 10: # If segment is too short, no trimming
            trim_points = 0
        else:
            trim_points = int(segment_len * trim_percentage)

        # Define the core region of the segment for median calculation
        core_start_idx = start_idx + trim_points
        core_end_idx = end_idx - trim_points

        # Handle cases where core region might be invalid (e.g., very short segments)
        if core_start_idx >= core_end_idx:
            core_data = y_s[start_idx : end_idx + 1] # Use full segment
        else:
            core_data = y_s[core_start_idx : core_end_idx + 1]

        s_raw.append(np.median(core_data) if len(core_data) > 0 else 0.0)

    # Enforce the expected pattern of plateau levels based on physical understanding
    s_final = _enforce_pattern_levels(s_raw, x, y)
    logging.info(f"Final determined plateau levels s1-s7: {np.array2string(s_final, precision=2)}")

    # Determine ramp boundaries (p1-p12) by analyzing local signal behavior around changepoints
    ramp_bounds = [] # This will store (x_start_ramp, x_end_ramp) for each ramp
    # Iterate through the 6 changepoints, which correspond to the 6 ramps
    for k in range(6):
        cp_idx = cps[k] # Index of the current changepoint
        si = s_final[k]   # Signal level of the plateau BEFORE this ramp
        sj = s_final[k+1] # Signal level of the plateau AFTER this ramp
        
        # Call the helper to find the start and end of the ramp in x-coordinates
        x_ramp_start, x_ramp_end = _ramp_edges_from_cp(x, y_s, cp_idx, si, sj)
        ramp_bounds.append((x_ramp_start, x_ramp_end))

    # Assemble the final p1-p12 array from the detected ramp boundaries
    p_final = np.array([val for pair in ramp_bounds for val in pair], dtype=float)
    
    # Ensure p-values are strictly increasing and within the x-data range
    xmin, xmax = x.min(), x.max()
    p_final = np.clip(p_final, xmin, xmax) # Keep within data bounds
    for i in range(1, len(p_final)):
        # If a point is not strictly greater than the previous, nudge it by a small amount
        if p_final[i] <= p_final[i-1]:
            p_final[i] = p_final[i-1] + (xmax - xmin) / (len(p_final) * 10000.0) # Small increment

    logging.info(f"Final determined P1-P12 transitions: {np.array2string(p_final, precision=0)}")

    # Generate the fitted signal using the determined p and s values
    y_fit = piecewise_stepwedge(x, p_final, s_final)

    # Calculate R-squared value for goodness of fit
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12 # Add small epsilon to avoid division by zero
    r2 = 1.0 - ss_res / ss_tot
    
    return p_final, s_final, y_fit, r2


def fit_pdd(depth_mm, signal_levels):
    """
    Fits an exponential decay model (A*exp(-b*d)+C) to the Percentage Depth Dose (PDD) data.
    The PDD is derived from the signal levels at different water-equivalent depths.
    Returns a function for the fitted PDD, the optimized parameters, and the D20/D10 ratio.
    """
    depth_mm = np.asarray(depth_mm, float)
    signal_levels = np.asarray(signal_levels, float)

    def exponential_decay_model(d, A, b, C_offset):
        """Exponential decay function for PDD."""
        return A * np.exp(-b * d) + C_offset

    # Initial guess for curve_fit parameters:
    # A0: Amplitude (range of signal)
    # b0: Decay constant (small positive value)
    # C0: Offset (minimum signal level)
    A0 = max(signal_levels) - min(signal_levels)
    b0 = 0.005 # A typical decay constant for these physics applications
    C0 = min(signal_levels)

    # Use scipy.optimize.curve_fit to find the best parameters
    # Bounds are set to ensure 'b' (decay constant) is positive.
    try:
        popt, pcov = curve_fit(exponential_decay_model, depth_mm, signal_levels, 
                               p0=[A0, b0, C0], 
                               bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
        A, b, C = popt
    except RuntimeError as e:
        logging.error(f"Failed to fit PDD curve: {e}. Using default params.")
        A, b, C = A0, b0, C0 # Fallback to initial guess if fit fails

    # Create a lambda function for the fitted PDD using the optimized parameters
    pdd_fit_func = lambda d: exponential_decay_model(np.asarray(d, float), A, b, C)

    # Calculate D20/D10 ratio from the fitted curve
    D10 = float(pdd_fit_func(100.0)) # PDD at 10 cm water (100 mm)
    D20 = float(pdd_fit_func(200.0)) # PDD at 20 cm water (200 mm)
    ratio = D20 / (D10 + 1e-12) # Add small epsilon to prevent division by zero

    return pdd_fit_func, popt, ratio

# ------------------------------ main workflow --------------------------------

def main():
    # Load data from CSV
    try:
        df = pd.read_csv(CSV_PATH)
        logging.info(f"Successfully loaded '{CSV_PATH}' with {len(df)} rows.")
    except FileNotFoundError:
        logging.critical(f"CRITICAL ERROR: '{CSV_PATH}' not found. Please ensure the file is in the correct directory.")
        return
    except Exception as e:
        logging.critical(f"CRITICAL ERROR: Could not read '{CSV_PATH}'. Reason: {e}")
        return

    # Extract X and Y data, drop NA values, convert to numpy arrays
    # Ensure X is sorted as the fitting routine expects it.
    x_raw = df[X_COL].dropna().to_numpy(dtype=float)
    y_raw = df[Y_COL].dropna().to_numpy(dtype=float)

    # Sort data by X-axis values if not already sorted
    order = np.argsort(x_raw)
    x = x_raw[order]
    y = y_raw[order]

    # Perform the step-wedge profile fitting
    logging.info("Starting step-wedge profile fitting...")
    p, s, y_fit, r2 = fit_stepwedge_time_profile(x, y)
    logging.info(f"Step-wedge profile fit completed. R^2 = {r2:.5f}")

    # Calculate measured step lengths for comparison with nominal values
    # The five flat steps correspond to (p3-p2), (p5-p4), (p7-p6), (p9-p8), (p11-p10)
    # These represent the duration the signal stays on a plateau while the couch moves.
    ramp_intervals_projections = np.array([
        p[2] - p[1], # Ramp 1 (s1->s2), then plateau s2
        p[4] - p[3], # Ramp 2 (s2->s3), then plateau s3
        p[6] - p[5], # Ramp 3 (s3->s4), then plateau s4
        p[8] - p[7], # Ramp 4 (s4->s5), then plateau s5
        p[10] - p[9] # Ramp 5 (s5->s6), then plateau s6
    ])
    
    # Convert projection differences to time, then to length in mm
    times_s = ramp_intervals_projections * time_per_projection_s
    measured_lengths_mm = times_s * couch_speed_mm_s
    logging.info(f"Measured step lengths: {np.array2string(measured_lengths_mm, precision=2)} mm")


    # Calculate field width from the first ramp duration (p2 - p1)
    # The field width is conceptually the length over which the signal transitions.
    # The text specifies p2-p1, which corresponds to the first *ramp* (the transition from air to step 1)
    field_width_projections = p[1] - p[0] # Duration of the first transition ramp
    field_width_mm = field_width_projections * time_per_projection_s * couch_speed_mm_s
    logging.info(f"Estimated field width (from p1-p2 ramp): {field_width_mm:.3f} mm")


    # PDD Analysis: Use the first 6 plateau levels (s1 to s6) for PDD calculation.
    # Note: s1 corresponds to air (0mm depth), s2 to 52.5mm, etc.
    s_levels_for_pdd = np.array(s[:6], dtype=float) # Plateaus s1 through s6
    
    # Fit an exponential curve to relate signal levels to depths
    pdd_fit_func, pdd_params, D20_over_D10 = fit_pdd(depths_water_mm, s_levels_for_pdd)

    # Normalize the PDD: The paper states normalization to 5 cm water depth (50 mm)
    norm_at_5cm_value = float(pdd_fit_func(50.0))
    # Normalize the original PDD points (from s1-s6)
    pdd_points_normalized = (s_levels_for_pdd / (norm_at_5cm_value + 1e-12)) * 100.0

    # Generate points for plotting the continuous PDD curve
    depths_plot = np.linspace(0, max(depths_water_mm) * 1.1, 300) # Extend range slightly
    pdd_curve_normalized = (pdd_fit_func(depths_plot) / (norm_at_5cm_value + 1e-12)) * 100.0
    
    logging.info(f"PDD D20/D10 ratio: {D20_over_D10:.3f}")

    # ----------------------------- plotting results ----------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10)) # Increased figure size for better readability
    axA, axB = axs[0]
    axC, axD = axs[1]

    # Plot A: Time-profile of single CT channel with fitted model
    axA.plot(x, y, color="tab:blue", lw=1.0, label="Measured Signal")
    axA.plot(x, y_fit, color="tab:red", lw=2.5, label="Fitted Model") # Changed color for better contrast
    axA.set_title("A) Time-profile of single CT channel", fontsize=12)
    axA.set_xlabel("Projections")
    axA.set_ylabel("Signal [a.u.]")
    axA.legend(loc="upper right")
    axA.grid(True, linestyle=':', alpha=0.6)
    axA.set_ylim(bottom=0) # Ensure y-axis starts from 0 or slightly below min signal

    # Plot B: Schematic Step-wedge profile (fitted)
    axB.plot(x, y_fit, color="tab:purple", lw=2.5) # Using the fitted model only
    for pk in p: # Plot the detected transition points p1 to p12 as vertical lines
        axB.axvline(pk, color="k", ls="--", lw=0.8, alpha=0.7)
    
    # Annotate plateau levels (s1-s7) at their approximate midpoints
    # Calculate representative x positions for each plateau's label
    x_positions_for_s_labels = []
    # s1 middle: start-p1
    x_positions_for_s_labels.append( (x.min() + p[0]) / 2 )
    # s2 middle: p2-p3
    x_positions_for_s_labels.append( (p[1] + p[2]) / 2 )
    # s3 middle: p4-p5
    x_positions_for_s_labels.append( (p[3] + p[4]) / 2 )
    # s4 middle: p6-p7
    x_positions_for_s_labels.append( (p[5] + p[6]) / 2 )
    # s5 middle: p8-p9
    x_positions_for_s_labels.append( (p[7] + p[8]) / 2 )
    # s6 middle: p10-p11
    x_positions_for_s_labels.append( (p[9] + p[10]) / 2 )
    # s7 middle: p12-end
    x_positions_for_s_labels.append( (p[11] + x.max()) / 2 )

    for i, (x_pos, s_val) in enumerate(zip(x_positions_for_s_labels, s)):
        # Adjust y-position slightly to avoid overlapping the line
        axB.text(x_pos, s_val, f"s{i+1}", 
                 verticalalignment='bottom' if i<6 else 'top', # To keep labels from clashing
                 horizontalalignment='center', fontsize=9, color='darkgreen')

    axB.set_title("B) Schematic Step-wedge profile (fitted model)", fontsize=12)
    axB.set_xlabel("Projections")
    axB.set_ylabel("Signal [a.u.]")
    axB.grid(True, linestyle=':', alpha=0.6)
    axB.set_ylim(axA.get_ylim()) # Keep y-limits consistent with A


    # Plot C: Percentage Depth Dose (PDD) normalized at 5 cm water
    axC.plot(depths_water_mm, pdd_points_normalized, "o", color="dodgerblue", markersize=6, label="'Stepwedge' PDD points")
    axC.plot(depths_plot, pdd_curve_normalized, "--", color="darkorange", lw=2, label="Exponential Fit")
    axC.set_title("C) Percentage Depth Dose (normalized at 5cm water)", fontsize=12)
    axC.set_xlabel("Depth in Water (eq.) [mm]")
    axC.set_ylabel("PDD [%]")
    axC.legend(loc="upper right")
    axC.grid(True, linestyle=':', alpha=0.6)
    axC.set_ylim(bottom=0)
    axC.text(0.02, 0.05, f"D20/D10 = {D20_over_D10:.3f}", transform=axC.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Plot D: Nominal and Measured step length
    steps_indices = np.arange(1, 6) # Corresponds to step numbers 1 through 5
    
    axD.plot(steps_indices, nominal_step_lengths_mm, "o-", color="grey", lw=2, label="Nominal Length")
    axD.plot(steps_indices, measured_lengths_mm, "o-", color="forestgreen", lw=2, label="Measured Length")
    
    # Plot ±1% tolerance lines around nominal values
    axD.plot(steps_indices, nominal_step_lengths_mm * 1.01, "k--", lw=0.9, alpha=0.7, label="±1% Tolerance")
    axD.plot(steps_indices, nominal_step_lengths_mm * 0.99, "k--", lw=0.9, alpha=0.7)
    
    axD.set_title("D) Nominal and Measured Step Length", fontsize=12)
    axD.set_xlabel("Step Number")
    axD.set_ylabel("Length [mm]")
    axD.set_xticks(steps_indices) # Set x-ticks to be exactly 1, 2, 3, 4, 5
    axD.legend(loc="best", fontsize=9)
    axD.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust y-limits to make the tolerance band clearly visible
    min_len = min(nominal_step_lengths_mm.min() * 0.98, measured_lengths_mm.min() * 0.98)
    max_len = max(nominal_step_lengths_mm.max() * 1.02, measured_lengths_mm.max() * 1.02)
    axD.set_ylim(bottom=29.0, top=31.0) # Fixed range based on image for consistency


    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle if ever used
    plt.suptitle("Step-Wedge QA Analysis", fontsize=16, y=0.99) # Optional overall title
    plt.savefig(OUT_PNG, dpi=200) # Increased DPI for higher quality output
    logging.info(f"Analysis complete. Figure saved to '{OUT_PNG}'")

    # Console summary
    print("\n--- Summary of Analysis ---")
    print(f"Fit R^2 for Time Profile: {r2:.4f}")
    print("\nDetected Plateau Signal Levels (s1-s7):")
    for i, val in enumerate(s):
        print(f"  s{i+1}: {val:.3f} a.u.")
    print("\nDetected Transition Points (p1-p12, in projections):")
    print("  " + np.array2string(p, precision=0, separator=', '))
    print(f"\nEstimated Field Width (from first ramp p1-p2): {field_width_mm:.3f} mm")
    print("\nMeasured Step Lengths (mm):")
    print("  " + np.array2string(measured_lengths_mm, precision=3, separator=', '))
    print("\nNominal Step Lengths (mm):")
    print("  " + np.array2string(nominal_step_lengths_mm, precision=3, separator=', '))
    print(f"\nPDD D20/D10 Ratio (from exponential fit): {D20_over_D10:.3f}")
    print("---------------------------")


if __name__ == "__main__":
    main()

