import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_charts(timestamps, dataset, similar_pairs, filename):
    # Calculate relative time from initial timestamp
    t = [t - timestamps[0] for t in timestamps]

    # Define the total number of charts required
    num_comparisons = len(similar_pairs)
    fig, axs = plt.subplots(num_comparisons, figsize=(10, 5 * num_comparisons), constrained_layout=True)
    
    # Adjust axs to be iterable, even if there's only one chart
    if num_comparisons == 1:
        axs = [axs]
    
    # Loop through the similar pairs to generate subplots
    for idx, ((param1, param2), title) in enumerate(similar_pairs.items()):
        axs[idx].set_title(title)
        
        # Extract data for both parameters
        data1 = dataset[param1]
        data2 = dataset[param2]
        
        # Plot the two parameters
        axs[idx].plot(t, data1, 'r', label=f"{param1}", linewidth=1)
        axs[idx].plot(t, data2, 'g', label=f"{param2}", linewidth=1)
        axs[idx].legend()
        
        # Calculate differences
        diffs = [b - a for a, b in zip(data1, data2)]
        d_min, d_max, d_avg = min(diffs), max(diffs), np.average(np.abs(diffs))
        
        # Add difference info as text in the subplot
        axs[idx].text(
            0.02, 0.95,
            f"min = {d_min:+}\nmax = {d_max:+}\nabs_avg = {d_avg:+.2f}",
            ha="left",
            va="top",
            transform=axs[idx].transAxes,
            bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8, 0.8), ec="none"),
            fontsize=10,
        )
        
        # Set labels
        axs[idx].set_ylabel('Value')
        if idx == num_comparisons - 1:
            axs[idx].set_xlabel('Time (s)')
    
    # Set the overall title for the entire figure
    fig.suptitle("Parameter Comparisons", fontsize=16)
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

# Usage example:
similar_pairs = {
    ##velocity
    ('GPS2_RAW.vel', 'GPS_RAW_INT.vel'): 'GPS-derived groundspeed comparison (cm/s)',
    ('GPS2_RAW.vel', 'VFR_HUD.groundspeed'): 'GPS vs VFR HUD groundspeed comparison (m/s)',
    ('GPS_RAW_INT.vel', 'VFR_HUD.groundspeed'): 'GPS vs VFR HUD groundspeed comparison (m/s)',
    ##altitude
    ('VFR_HUD.alt', 'AHRS3.altitude'): 'Altitude comparison (meters)',
    ('GPS_RAW_INT.alt', 'GPS2_RAW.alt'): 'GPS RAW vs GPS2 RAW altitude comparison (millimeters)',
    ('VFR_HUD.alt', 'GPS_RAW_INT.alt'): 'VFR HUD vs GPS RAW altitude comparison (meters to millimeters)',
    ('VFR_HUD.alt', 'GPS2_RAW.alt'): 'VFR HUD vs GPS2 RAW altitude comparison (meters to millimeters)',
    ('AHRS3.altitude', 'GPS_RAW_INT.alt'): 'AHRS3 vs GPS RAW altitude comparison (meters to millimeters)',
    ('AHRS3.altitude', 'GPS2_RAW.alt'): 'AHRS3 vs GPS2 RAW altitude comparison (meters to millimeters)',
    ##HDOP
    ('GPS2_RAW.eph', 'GPS_RAW_INT.eph'): 'GPS2 RAW vs GPS RAW HDOP comparison',
    ##satellites
    ('GPS2_RAW.satellites_visible', 'GPS_RAW_INT.satellites_visible'): 'GPS2 RAW vs GPS RAW visible satellites comparison'
}

# Sample function call:
# generate_comparison_charts(timestamps, dataset, similar_pairs, 'output_filename.png')
