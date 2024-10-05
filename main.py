#!/usr/bin/env python3
import os
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from telemetry_parser import parse_telemetry


def parse_args():
    """
    This script takes a tlog file and extract some data
    """
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "-t",
        "--tlog",
        required=True,
        help="Tlog filename or path to parse",
    )
    parser.add_argument(
        "--head",
        type=int,
        required=False,
        default=-1,
        nargs="?",
        help="process only the first N entries (if included, defaults to 10)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="a Tlog file gets cached. Pass this flag processing all over again",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="output filename (defaults to something based on the tlog name",
    )

    parsed = parser.parse_args()
    if parsed.head is None:
        parsed.head = 10  # flag was included, so default is applied

    return parsed

def save2csv(filename, timestamps, dataset, units):
    print(f"Saving data to {filename} (takes some time)")
    np.savetxt(
        f"{filename}",
        np.transpose([timestamps, *dataset.values()]),
        delimiter=",",
        fmt="%s",
        header=",".join(["timestamp", *dataset.keys()]),
        comments="",
    )


def generate_comparison_charts(timestamps, dataset, units, similar_pairs, filename_prefix):
    if not timestamps or not dataset:
        print("Error: Timestamps or dataset is empty. Please check the telemetry parsing.")
        return

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
        print(f"Generating chart for: {param1} vs {param2}")

        # Extract data for both parameters
        data1 = dataset.get(param1)
        data2 = dataset.get(param2)

        if data1 is None or data2 is None:
            print(f"Warning: Missing data for parameters {param1} or {param2}. Skipping this comparison.")
            continue

        # Plot the two parameters
        axs[idx].set_title(title)
        axs[idx].plot(t, data1, 'r', label=f"{param1}", linewidth=1)
        axs[idx].plot(t, data2, 'g', label=f"{param2}", linewidth=1)
        axs[idx].legend()

        # Calculate differences
        diffs = [b - a for a, b in zip(data1, data2)]
        d_min, d_max, d_avg = min(diffs), max(diffs), np.average(np.abs(diffs))

        # Add difference info as text in the subplot
        axs[idx].text(
            0.02, 0.95,
            f"min = {round(d_min,2):+}\nmax = {round(d_max,2):+}\nabs_avg = {round(d_avg,2):+}",
            ha="left",
            va="top",
            transform=axs[idx].transAxes,
            bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8, 0.8), ec="none"),
            fontsize=10,
        )

        # Set labels
        axs[idx].set_ylabel(f"{units[param1]} | {units[param2]}")
        axs[idx].set_xlabel('Time (s)')

    # Set the overall title for the entire figure
    fig.suptitle("Parameter Comparisons", fontsize=16)

    # Save the figure
    filename = f"{filename_prefix}-comparison.png"
    print(f"Saving chart to {filename}")
    fig.savefig(filename)
    plt.close()


def main() -> int:
    args = parse_args()
    timestamps, dataset, units = [], {}, {}

    cache_filename = f"{args.tlog}.h{args.head}.pickle"
    if not args.no_cache and os.path.isfile(cache_filename):
        print("Using cached values already processed for this tlog file")
        with open(cache_filename, "rb") as f:
            timestamps, dataset, units = pickle.load(f)
    else:
        print("Parsing telemetry data from tlog...")
        timestamps, dataset, units = parse_telemetry(
            args.tlog,
            fields = [
                "VFR_HUD.heading",
                "VFR_HUD.alt",
                "VFR_HUD.climb",
                "VFR_HUD.groundspeed",
                "VFR_HUD.airspeed",
                "VFR_HUD.throttle",
                "GLOBAL_POSITION_INT.vx",
                "GPS2_RAW.eph",
                "GPS_RAW_INT.eph",
                "GPS2_RAW.vel",
                "GPS_RAW_INT.vel",
                "AHRS3.altitude",
                "GPS2_RAW.alt",
                "GPS_RAW_INT.alt",
                "GPS2_RAW.satellites_visible",
                "GPS_RAW_INT.satellites_visible",
            ],
            head=args.head,
        )
        print("Caching parsed data...")
        with open(cache_filename, "wb") as f:
            pickle.dump([timestamps, dataset, units], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Validate dataset contents
    if not timestamps or not dataset:
        print("Error: No data available after parsing the telemetry file.")
        return 1

    # Print dataset keys
    print(f"Available keys in dataset: {list(dataset.keys())}")

    # Print sample values for each key in the dataset
    for key in dataset:
        print(f"Key: {key}, Length: {len(dataset[key])}, Sample: {dataset[key][:5]}")

    # Check which pairs are not available in the dataset
    similar_pairs = {
        ##velocity
        ('GPS2_RAW.vel', 'GPS_RAW_INT.vel'): 'GPS-derived groundspeed comparison',
        ('GPS2_RAW.vel', 'VFR_HUD.groundspeed'): 'GPS vs VFR HUD groundspeed comparison',
        ('GPS_RAW_INT.vel', 'VFR_HUD.groundspeed'): 'GPS vs VFR HUD groundspeed comparison',
        ##altitude
        ('VFR_HUD.alt', 'AHRS3.altitude'): 'Altitude comparison',
        ('GPS_RAW_INT.alt', 'GPS2_RAW.alt'): 'GPS RAW vs GPS2 RAW altitude comparison',
        ('VFR_HUD.alt', 'GPS_RAW_INT.alt'): 'VFR HUD vs GPS RAW altitude comparison',
        ('VFR_HUD.alt', 'GPS2_RAW.alt'): 'VFR HUD vs GPS2 RAW altitude comparison',
        ('AHRS3.altitude', 'GPS_RAW_INT.alt'): 'AHRS3 vs GPS RAW altitude comparison',
        ('AHRS3.altitude', 'GPS2_RAW.alt'): 'AHRS3 vs GPS2 RAW altitude comparison',
        ##HDOP
        ('GPS2_RAW.eph', 'GPS_RAW_INT.eph'): 'GPS2 RAW vs GPS RAW HDOP comparison',
        ##satellites
        ('GPS2_RAW.satellites_visible', 'GPS_RAW_INT.satellites_visible'): 'GPS2 RAW vs GPS RAW visible satellites comparison'
    }

    # Check availability of each pair in dataset
    for (param1, param2), title in similar_pairs.items():
        if param1 not in dataset or param2 not in dataset:
            print(f"Warning: Data missing for pair: {param1}, {param2}")

    # Normalize units
    for field in dataset:
        if not units[field]:
            continue
        if units[field].startswith("cm"):
            dataset[field] = np.divide(dataset[field], 100)
            units[field] = units[field][1:]
        elif units[field].startswith("mm"):
            dataset[field] = np.divide(dataset[field], 1000)
            units[field] = units[field][1:]

    # save raw data to csv
    save2csv(f"{args.tlog}.csv", timestamps, dataset, units)

    # Create output directory for figures
    os.makedirs(f"{args.tlog}-figs", exist_ok=True)
    generate_comparison_charts(timestamps, dataset, units, similar_pairs, f"{args.tlog}-figs/fig01")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
