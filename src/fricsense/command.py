import crc
import time

import numpy as np


command_dt = np.dtype(
    [
        ("timestamp", np.uint32),
        ("command", np.uint32),
        ("payload", np.uint8, 18),
        ("crc", np.uint16),
    ]
)


def make_command(command, payload=None):
    crc_ = crc.Calculator(crc.Crc16.KERMIT, optimized=True)

    cmd = np.zeros(1, dtype=command_dt)[0]

    # Arbitrary offset:
    # Mon Oct 28 2024 12:00:00 GMT-0400 (Eastern Daylight Time)
    cmd["timestamp"] = np.uint32(int(time.time()) - 1730131200)
    cmd["command"] = command
    if payload is not None:
        if not isinstance(payload, bytes):
            raise ValueError("Payload must be bytes")
        if len(payload) > len(payload):
            raise ValueError("Payload too large")
        cmd["payload"][: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    cmd["crc"] = crc_.checksum(cmd.tobytes()[:-2])

    return cmd.tobytes()
