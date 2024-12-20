from pymavlink import mavutil, DFReader


def parse_fields(fields: list[str]) -> dict[str, set]:
    msg_types = {}
    for field in fields:
        typ, fiel = field.split(".")
        if typ not in msg_types:
            msg_types[typ] = set()

        msg_types[typ].add(fiel)
    return msg_types


def parse_telemetry(
    tlog_file: str, fields: list[str], head=-1
) -> tuple[list[float], dict[str, list], dict[str, str]]:
    mlog = mavutil.mavlink_connection(
        tlog_file,
        dialect="ardupilotmega",
        progress_callback=lambda x: print(f"process tlog... {x/100:.0%}\r", end=""),
    )
    print(f"processed tlog ({mlog.data_len} entries)\r")

    n_entries = mlog.data_len if head <= 0 else head

    print("Allocating arrays")
    msg_types = parse_fields(fields)
    tstamps = [None] * n_entries
    dataset = {field: [None] * n_entries for field in fields}

    i=0
    for i in range(n_entries):
        msg = mlog.recv_match(type=set(msg_types.keys()))
        if msg is None:
            break
        if i % 1000 == 0:
            print(
                f"Parsing relevant fields ({(mlog.offset if head <= 0 else i)*100/n_entries:.1f}%)\r",
                end="",
            )

        if hasattr(msg, "msgname") and msg.msgname in msg_types:
            tstamps[i] = getattr(msg, "_timestamp", None)
            for fname in msg_types[msg.msgname]:
                dataset[f"{msg.msgname}.{fname}"][i] = getattr(msg, fname, None)
    print("Parsed all relevant fields         ")

    print("Cleaning unused fields")
    # remove timstamps without data
    tstamps = tstamps[:i]
    for k in dataset:
        dataset[k] = dataset[k][:i]

    print("Remove None values")
    # fill None values - the easy way
    for i in range(1, len(tstamps)):
        for field in dataset:
            if dataset[field][i] is None:
                dataset[field][i] = dataset[field][i - 1]
    for field in dataset:  # fill values before first None
        for j, v in enumerate(dataset[field]):
            if v is not None:
                dataset[field][:j] = [v] * (j)
                break

    units = {}
    for typ in mlog.messages:
        for field in getattr(mlog.messages[typ], "fieldnames", []):
            unit = getattr(mlog.messages[typ], "fieldunits_by_name", {}).get(field, "-")
            units[f"{typ}.{field}"] = unit

    return tstamps, dataset, units

def parse_telemetry_bin(
    bin_file: str, fields: list[str], head=-1
) -> tuple[list[float], dict[str, list], dict[str, str]]:
    mlog = DFReader.DFReader_binary(
        bin_file,
        progress_callback=lambda x: print(f"process bin... {x/100:.0%}\r", end=""),
    )
    print(f"processed bin ({mlog.data_len} entries)\r")

    n_entries = mlog.data_len if head <= 0 else head

    print("Allocating arrays")
    msg_types = parse_fields(fields)
    tstamps = []
    dataset = {field: [] for field in fields}

    i=0
    for i in range(n_entries):
        msg = mlog.recv_match(type=set(msg_types.keys()))
        # msg = mlog.recv_msg()
        if msg is None:
            break
        if i % 1000 == 0:
            print(
                f"Parsing relevant fields ({(mlog.offset if head <= 0 else i)*100/n_entries:.1f}%)\r",
                end="",
            )

        tstamps.append(getattr(msg, "_timestamp", None))
        inserted=set()
        for fname in msg_types[msg.get_type()]:
            inserted.add(f"{msg.get_type()}.{fname}")
            dataset[f"{msg.get_type()}.{fname}"].append(getattr(msg, fname, None))
        for field in set(dataset.keys())-inserted:
            dataset[field].append(None)

    print("Parsed all relevant fields         ")

    print("Cleaning unused fields")
    # remove timstamps without data
    tstamps = tstamps[:i]
    for k in dataset:
        dataset[k] = dataset[k][:i]

    print("Remove None values")
    # fill None values - the easy way
    for i in range(1, len(tstamps)):
        for field in dataset:
            if dataset[field][i] is None:
                dataset[field][i] = dataset[field][i - 1]
    for field in dataset:  # fill values before first None
        for j, v in enumerate(dataset[field]):
            if v is not None:
                dataset[field][:j] = [v] * (j)
                break

    units = {}
    for typ in mlog.messages:
        for field in getattr(mlog.messages[typ], "_fieldnames", []):
            unit = getattr(mlog.messages[typ], "fieldunits_by_name", {}).get(field, "-")
            units[f"{typ}.{field}"] = unit

    return tstamps, dataset, units
