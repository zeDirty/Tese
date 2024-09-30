#!/usr/bin/env python3
from argparse import ArgumentParser
from telemetry_parser import parse_telemetry


def parse_args():
    """
    This script takes a tlog file and extract some data
    """
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "command",
        help="what to do",
        choices=[
            "tlog2csv",
            "diffs",
        ],
    )
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
    telemetry = parse_telemetry(
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
    )
    print(telemetry)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
