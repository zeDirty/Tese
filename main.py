#!/usr/bin/env python3
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from telemetry_parser import parse_telemetry, parse_telemetry_bin


def parse_args():
    """
    This script takes a tlog file and extract some data
    """
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "-t",
        "--tlog",
        required=False,
        help="Tlog filename or path to parse",
    )
    parser.add_argument(
        "-b",
        "--bin",
        required=False,
        help="Bin filename or path to parse",
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
    readable_timestamp = [datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]

    np.savetxt(
        f"{filename}",
        np.transpose([timestamps, readable_timestamp, *dataset.values()]),
        delimiter=",",
        fmt="%s",
        header=",".join(["timestamp", "datetime", *dataset.keys()]),
        comments="",
    )


def generate_comparison_charts(timestamps, dataset, units, similar_pairs, anomalies, filename_prefix):
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
        anomalies1 = anomalies.get(param1,[])
        anomalies2 = anomalies.get(param2,[])

        if data1 is None or data2 is None:
            # print(f"Warning: Missing data for parameters {param1} or {param2}. Skipping this comparison.")
            continue

        # Plot the two parameters
        axs[idx].set_title(title)
        axs[idx].plot(t, anomalies1, 'darkred', label=f"anomalies {param1}", linewidth=5)
        axs[idx].plot(t, anomalies2, 'darkgreen', label=f"anomalies {param2}", linewidth=5)
        axs[idx].plot(t, data1, 'red', label=f"{param1}", linewidth=1)
        axs[idx].plot(t, data2, 'mediumseagreen', label=f"{param2}", linewidth=1)
        axs[idx].legend()

        # Calculate differences
        diffs = [b - a for a, b in zip(data1, data2)]
        d_min, d_max, d_avg, d_aavg = min(diffs), max(diffs), np.average(diffs), np.average(np.abs(diffs))

        # Add difference info as text in the subplot
        axs[idx].text(
            0.02, 0.95,
            f"min = {round(d_min,2):+}\nmax = {round(d_max,2):+}\navg = {round(d_avg,2):+}\nabs_avg = {round(d_aavg,2):+}",
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

    input_filename=""
    if not args.tlog and not args.bin:
        print("--bin or --tlog is mandatory")
        return 1
    if args.tlog and args.bin:
        print("--bin and --tlog are mutually exclusive")
        return 1

    if args.tlog:
        input_filename=args.tlog
        cache_filename = f"{input_filename}.h{args.head}.pickle"
        if not args.no_cache and os.path.isfile(cache_filename):
            print("Using cached values already processed for this tlog file")
            with open(cache_filename, "rb") as f:
                timestamps, dataset, units = pickle.load(f)
        else:
            print("Parsing telemetry data from tlog...")
            timestamps, dataset, units = parse_telemetry(
                input_filename,
                fields = [
                    "AIRSPEED_AUTOCAL.vx", 
                    "AIRSPEED_AUTOCAL.vy",  
                    "AIRSPEED_AUTOCAL.vz",  
                    "GLOBAL_POSITION_INT.vx",  
                    "GLOBAL_POSITION_INT.vy",  
                    "GLOBAL_POSITION_INT.vz",  
                    "LOCAL_POSITION_NED.vx",  
                    "LOCAL_POSITION_NED.vy",  
                    "LOCAL_POSITION_NED.vz",  
                    "VFR_HUD.heading",
                    "VFR_HUD.alt",
                    "VFR_HUD.climb",
                    "VFR_HUD.groundspeed",
                    "VFR_HUD.airspeed",
                    "VFR_HUD.throttle",
                    "GLOBAL_POSITION_INT.vx",
                    "GPS2_RAW.vel",
                    "GPS_RAW_INT.vel",
                    "AHRS3.altitude",
                    "GPS2_RAW.alt",
                    "GPS_RAW_INT.alt",
                    "GPS2_RAW.satellites_visible",
                    "GPS_RAW_INT.satellites_visible",
                    "SCALED_PRESSURE.temperature",
                    "SENSOR_OFFSETS.raw_temp",
                    "RAW_IMU.xacc",
                    "RAW_IMU.yacc",
                    "RAW_IMU.zacc", 
                    "SCALED_IMU2.xacc", 
                    "SCALED_IMU2.yacc",
                    "SCALED_IMU2.zacc"
                ],
                head=args.head,
            )
            print("Caching parsed data...")
            with open(cache_filename, "wb") as f:
                pickle.dump([timestamps, dataset, units], f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.bin:
        input_filename=args.bin
        cache_filename = f"{input_filename}.h{args.head}.bin.pickle"
        if not args.no_cache and os.path.isfile(cache_filename):
            print("Using cached values already processed for this bin file")
            with open(cache_filename, "rb") as f:
                timestamps, dataset, units = pickle.load(f)
        else:
            print("Parsing telemetry data from bin...")
            timestamps, dataset, units = parse_telemetry_bin(
                input_filename,
                fields = [
                    "IMU.AccX",
                    "IMU.AccY",
                    "IMU.AccZ",
                    "IMU2.AccX",
                    "IMU2.AccY",
                    "IMU2.AccZ",
                    "STAT.isFlying",
                    "STAT.isFlyProb",
                    "STAT.Armed",
                    "STAT.Stage",
                    "GPS.HDop",
                    "GPS2.HDop",
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

    similar_pairs_bin = {
    }

    # Check which pairs are not available in the dataset
    similar_pairs = {
        ##velocity
        ('AIRSPEED_AUTOCAL.vx', 'GLOBAL_POSITION_INT.vx'): 'AUTOCAL vs GLOBAL_POSITION x-direction',
        ('AIRSPEED_AUTOCAL.vx', 'LOCAL_POSITION_NED.vx'): 'AUTOCAL vs LOCAL_POSITION x-direction',
        ('GLOBAL_POSITION_INT.vx', 'LOCAL_POSITION_NED.vx'): 'GLOBAL_POSITION vs LOCAL_POSITION x-direction',
        ('AIRSPEED_AUTOCAL.vy', 'GLOBAL_POSITION_INT.vy'): 'AUTOCAL vs GLOBAL_POSITION y-direction',
        ('AIRSPEED_AUTOCAL.vy', 'LOCAL_POSITION_NED.vy'): 'AUTOCAL vs LOCAL_POSITION y-direction',
        ('GLOBAL_POSITION_INT.vy', 'LOCAL_POSITION_NED.vy'): 'GLOBAL_POSITION vs LOCAL_POSITION y-direction',
        ('AIRSPEED_AUTOCAL.vz', 'GLOBAL_POSITION_INT.vz'): 'AUTOCAL vs GLOBAL_POSITION z-direction',
        ('AIRSPEED_AUTOCAL.vz', 'LOCAL_POSITION_NED.vz'): 'AUTOCAL vs LOCAL_POSITION z-direction',
        ('GLOBAL_POSITION_INT.vz', 'LOCAL_POSITION_NED.vz'): 'GLOBAL_POSITION vs LOCAL_POSITION z-direction',
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
        ##aceleracao
        ('RAW_IMU.xacc', 'SCALED_IMU2.xacc'): 'IMU Raw vs IMU2 Raw X acelaration comparison',
        ('RAW_IMU.yacc', 'SCALED_IMU2.yacc'): 'IMU Raw vs IMU2 Raw Y acelaration comparison',
        ('RAW_IMU.zacc', 'SCALED_IMU2.zacc'): 'IMU Raw vs IMU2 Raw Z acelaration comparison',
        ##HDOP
        ('GPS2_RAW.eph', 'GPS_RAW_INT.eph'): 'GPS2 RAW vs GPS RAW HDOP comparison',
        ##satellites
        ('GPS2_RAW.satellites_visible', 'GPS_RAW_INT.satellites_visible'): 'GPS2 RAW vs GPS RAW visible satellites comparison',
        ##Temperaturas
        ('SCALED_PRESSURE.temperature', 'SENSOR_OFFSETS.raw_temp'): 'Temperatures',


        ##acceleration
        ('IMU.AccX', 'IMU2.AccX'): 'IMU vs IMU2 X acelaration comparison',
        ('IMU.AccY', 'IMU2.AccY'): 'IMU vs IMU2 Y acelaration comparison',
        ('IMU.AccZ', 'IMU2.AccZ'): 'IMU vs IMU2 Z acelaration comparison',
        ##HDOP
        ('GPS.HDop', 'GPS2.HDop'): 'GPS vs GPS2 HDOP comparison',

    }

    anomaly_thresholds_bin = {
    }

    anomaly_thresholds = {
        # SPEED AIRSPEED_AUTOCAL
        'AIRSPEED_AUTOCAL.vx': (-39.098, 39.098),  #x-direction (m/s) -29.15 33.16 (mean 0.08)
        'AIRSPEED_AUTOCAL.vy': (-39.098, 39.098),  #y-direction (m/s) -28.26 33.19 mean 0.01
        'AIRSPEED_AUTOCAL.vz': (-39.098, 39.098),  #z-direction (m/s) -5.69 5.88 mean 0

        # SPEED GLOBAL_POSITION_INT
        'GLOBAL_POSITION_INT.vx': (-3909.8, 3909.8),  #x-direction (cm/s) -2946 3312 mean 5.81
        'GLOBAL_POSITION_INT.vy': (-3909.8, 3909.8),  #y-direction (cm/s) -2841 3332 mean 1.35
        'GLOBAL_POSITION_INT.vz': (-3909.8, 3909.8),  #z-direction (cm/s) -670 577 mean 0.45

        # SPEED LOCAL_POSITION_NED
        'LOCAL_POSITION_NED.vx': (-39.098, 39.098),  #x-direction (m/s) -29.47 33.13 mean 0.06
        'LOCAL_POSITION_NED.vy': (-39.098, 39.098),  #y-direction (m/s) -28.42 33.33 mean 0.02
        'LOCAL_POSITION_NED.vz': (-39.098, 39.098),  #z-direction (m/s) -6.71 5.78 mean 0

        # SPEED GPS2_RAW and GPS_RAW_INT
        'GPS2_RAW.vel': (21.606, 39.098),  #Threshold for GPS-derived groundspeed (cm/s) similar to: GPS_RAW_INT.vel and VFR_HUD.groundspeed
        'GPS_RAW_INT.vel': (21.606, 39.098),  #Threshold for GPS-derived groundspeed (cm/s) similar to: GPS2_RAW.vel and VFR_HUD.groundspeed

        # SPEED VFR_HUD
        'VFR_HUD.airspeed': (21.606, 39.098),  #42 and 76 knots 21.606, 39.098 (m/s)
        'VFR_HUD.groundspeed': (21.606, 39.098),  #(m/s) similar to: GPS_RAW_INT.vel and GPS2_RAW Accelerometers Threshold x-axis (0 to 3.8 g) verificar unidades??? -11500 e 500

        # Accelerometers RAW_IMU
        'RAW_IMU.xacc': (-980, 980),  # Threshold x-axis (0 to 3.8 g) verificar unidades??? -11500 e 500
        'RAW_IMU.yacc': (-980, 980),  # Threshold y-axis (0 to 3.8 g) -1000 e 3000
        'RAW_IMU.zacc': (-1960, 0),  # Threshold z-axis (0 to 3.8 g) -1600 e 200

        # Accelerometers SCALED_IMU2
        'SCALED_IMU2.xacc': (-980, 980),  # Threshold x-axis (0 to 3.8 g) (mG) valores -380 a 501 (spike no final de -11000)
        'SCALED_IMU2.yacc': (-980, 980),  # Threshold y-axis (0 to 3.8 g) (mG) -933 a 1177  (spike no final de 3333)
        'SCALED_IMU2.zacc': (-1960, 0),  # Threshold z-axis (0 to 3.8 g) (mG) -1533 a -361

        # ALTITUDE
        'VFR_HUD.alt': (0, 3078),  # Altitude between 0 and 3078 meters (SIMILAR A AHRS3.altitude)
        'AHRS3.altitude': (0, 3078),  # Absolute altitude in meters (SIMILAR A VFR_HUD.alt)
        'GPS_RAW_INT.alt': (0, 3078),  # Altitude in m (SIMILAR A GPS2_RAW)
        'GPS2_RAW.alt': (0, 3078),  # Absolute altitude in milimeters (SIMILAR A GPS_RAW_INT)
        'GLOBAL_POSITION_INT.alt': (0, 3078),  # Altitude in milimeters
        'GLOBAL_POSITION_INT.relative_alt': (0, 3078),  # Relative altitude in milimeters
        'LOCAL_POSITION_NED.z': (-3078, 0),  # Altitude in local frame

        # HDOP (Horizontal Dilution of Precision)
        #'GPS2_RAW.eph': (0, 10),  # GPS2_RAW eph gps2 raw_t(Min:60 Max: 139 Mean: 75.83)
        #'GPS_RAW_INT.eph': (0, 10),  # GPS_RAW_INT eph gps raw int t(Min:60 Max:142 Mean:75.42)

        #visible satellites
        'GPS2_RAW.satellites_visible': (5, 20),  # Number of satellites visible in GPS2_RAW (0 to 20)
        'GPS_RAW_INT.satellites_visible': (5, 20),  # Number of satellites visible in GPS_RAW_INT (0 to 20)

        #Temperaturas
        'SCALED_PRESSURE.temperature': (4000, 5600),  # cdegC similares (3998; 5558) 4293.4 equivale em Cº (39,98; 55,58) 42,934
        'SENSOR_OFFSETS.raw_temp': (4000, 5600),  # cdegC similares (4000; 5558) media: 4293.18

        #acceleration
        "IMU.AccX": (-9.8, 9.8), 
        "IMU.AccY": (-9.8, 9.8), 
        "IMU.AccZ": (-19.6, 0),
        "IMU2.AccX": (-9.8, 9.8),
        "IMU2.AccY": (-9.8, 9.8),
        "IMU2.AccZ": (-19.6, 0),
        #HDOP
        "GPS.HDop": (0, 10),
        "GPS2.HDop": (0, 10),

    }

    # # Check availability of each pair in dataset
    # for (param1, param2), title in similar_pairs.items():
    #     if param1 not in dataset or param2 not in dataset:
    #         print(f"Warning: Data missing for pair: {param1}, {param2}")

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


    # Check anomalies
    print("Checking anomalies")
    anomalies = {}
    for anm, th in anomaly_thresholds.items():
        if anm not in dataset:
            # print(f"{anm} not captured in dataset")
            continue
        if len(th) != 2:
            # print(f"Invalid threshold for {anm}: {th}")
            continue
        anomalies[anm] = [ x if x<th[0] or x>th[1] else None for x in dataset[anm]]

    # save raw data to csv
    save2csv(f"{input_filename}.csv", timestamps, dataset, units)
    save2csv(f"{input_filename}-anomalies.csv", timestamps, anomalies, units)

    # save fields units to file
    np.savetxt(f"{input_filename}.units.txt", [ f"{_f}, {_u}" for _f,_u in units.items() ], fmt="%s",)

    # Create output directory for figures
    os.makedirs(f"{input_filename}-figs", exist_ok=True)
    generate_comparison_charts(timestamps, dataset, units, similar_pairs, anomalies, f"{input_filename}-figs/fig01")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
