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


def generate_gps2_charts(timestamps, dataset, filename):
    fig, axs = plt.subplots(2, figsize=(8, 8), layout="constrained")
    fig.suptitle("Compare GPS2_RAW.eph and GPS_RAW_INT.eph")

    t = [t - timestamps[0] for t in timestamps]
    axs[0].plot(t, dataset["GPS2_RAW.eph"], "r", label=["GPS2_RAW.eph"], linewidth=1)
    axs[0].plot(
        t, dataset["GPS_RAW_INT.eph"], "g", label=["GPS_RAW_INT.eph"], linewidth=1
    )
    axs[0].legend()
    axs[0].set_ylabel("heigh (m)")

    diffs = [b - a for a, b in zip(dataset["GPS2_RAW.eph"], dataset["GPS_RAW_INT.eph"])]
    d_min, d_max, d_avg = min(diffs), max(diffs), np.average(np.abs(diffs))
    axs[1].plot(t, diffs, "r", label=["diffs.eph"], linewidth=1)
    axs[1].set_ylabel("error (m)")
    axs[1].set_xlabel("time (s)")
    axs[1].legend()
    axs[1].text(
        0.02,
        0.95,
        f"min = {d_min:+}\nmax = {d_max:+}\nabs_avg = {d_avg:+.2}",
        ha="left",
        va="top",
        transform=axs[1].transAxes,
        bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8, 0.8), ec="none"),
        fontsize=12,
    )

    print(f"Saving chart to {filename}")
    fig.savefig(filename)


def main() -> int:
    args = parse_args()
    timestamps, dataset = [], {}

    cache_filename = f"{args.tlog}.h{args.head}.pickle"
    if not args.no_cache and os.path.isfile(cache_filename):
        print("Using cached values already processed for this tlog file")
        with open(cache_filename, "rb") as f:
            timestamps, dataset = pickle.load(f)
    else:
        timestamps, dataset = parse_telemetry(
            args.tlog,
            fields=[
                "VFR_HUD.heading",
                "VFR_HUD.alt",
                "VFR_HUD.climb",
                "VFR_HUD.groundspeed",
                "VFR_HUD.airspeed",
                "VFR_HUD.throttle",
                "GLOBAL_POSITION_INT.vx",
                "GPS2_RAW.eph",
                "GPS_RAW_INT.eph",
            ],
            head=args.head,
        )
        with open(cache_filename, "wb") as f:
            pickle.dump([timestamps, dataset], f, protocol=pickle.HIGHEST_PROTOCOL)

    # save raw data to csv
    print(f"Saving data to {args.tlog}.csv (takes some time)")
    np.savetxt(
        f"{args.tlog}.csv",
        np.transpose([timestamps, *dataset.values()]),
        delimiter=",",
        fmt="%s",
        header=",".join(["timestamp", *dataset.keys()]),
        comments="",
    )

    os.makedirs(f"{args.tlog}-figs", exist_ok=True)
    generate_gps2_charts(timestamps, dataset, f"{args.tlog}-figs/fig01.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
