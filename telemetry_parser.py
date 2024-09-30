from pymavlink import mavutil


def parse_fields(fields: list[str]) -> dict[str, set]:
    msg_types = {}
    for field in fields:
        typ, fiel = field.split(".")
        if typ not in msg_types:
            msg_types[typ] = set()

        msg_types[typ].add(fiel)
    return msg_types


def parse_telemetry(tlog_file: str, fields: list[str]):
    mlog = mavutil.mavlink_connection(
        tlog_file,
        dialect="ardupilotmega",
        progress_callback=lambda x: print(f"process tlog... {x/100:.0%}\r", end=""),
    )

    msg_types = parse_fields(fields)
    print(msg_types)

    i = 1
    while msg := mlog.recv_msg():
        print(i, msg)
        i += 1
        if i > 20:
            break

    return mlog
