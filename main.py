#!/usr/bin/env python3
from argparse import ArgumentParser
from telemetry_parser import parse_telemetry
import matplotlib.pyplot as plt
import numpy as np


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
        "-o",
        "--output",
        required=False,
        help="output filename (defaults to something based on the tlog name",
    )

    parsed = parser.parse_args()
    if parsed.head is None:
        parsed.head = 10  # flag was included, so default is applied

    return parsed


def main() -> int:
    args = parse_args()
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

    print("Showing a chart")
    t = [t-timestamps[0] for t in timestamps]
    plt.plot(t, dataset["GPS2_RAW.eph"], "r", label=["GPS2_RAW.eph"])
    plt.plot(t, dataset["GPS_RAW_INT.eph"], "g", label=["GPS_RAW_INT.eph"])
    plt.xlabel("seconds")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
