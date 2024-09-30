from pymavlink import mavutil


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
) -> tuple[list[float], dict[str, list]]:
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

    valid_idx = []
    for i in range(n_entries):
        if i % 10000 == 0:
            print(f"Parsing relevant fields ({i*100/n_entries:.1f}%)\r", end="")
        msg = mlog.recv_msg()

        if hasattr(msg, "msgname") and msg.msgname in msg_types:
            tstamps[i] = getattr(msg, "_timestamp", None)
            valid_idx.append(i)
            for fname in msg_types[msg.msgname]:
                dataset[f"{msg.msgname}.{fname}"][i] = getattr(msg, fname, None)
    print("Parsed all relevant fields         ")

    print("Cleaning unused fields")
    # remove timstamps without data
    new_tstamps = [None] * len(valid_idx)
    new_dataset = {field: [None] * len(valid_idx) for field in fields}
    for i, idx in enumerate(valid_idx):
        new_tstamps[i] = tstamps[idx]
    for key in new_dataset:
        for i, idx in enumerate(valid_idx):
            new_dataset[key][i] = dataset[key][idx]
    tstamps, dataset = new_tstamps, new_dataset

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

    return tstamps, dataset
