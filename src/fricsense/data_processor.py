import crc
import numpy as np
import pandas as pd
import time

from enum import Enum
from typing import List, Tuple, Optional

from .ring_buffer import RingBuffer


class InvalidHeaderChecksum(Exception):
    pass


class InvalidDataChecksum(Exception):
    pass


class UnknownPacketType(Exception):
    pass


class PacketType(Enum):
    Audio = 0xAA  # 170
    TED = 0xCC  # 204
    IMU = 0xDD  # 221
    Env = 0xEE  # 238
    Mag = 0xFF  # 255


header_dt = np.dtype(
    [
        ("syncword", "<u4"),
        ("timestamp_ms", "<u4"),
        ("data_size", "<u2"),
        ("data_type", "<u2"),
        ("data_crc", "<u2"),
        ("header_crc", "<u2"),
    ]
)

imu_dt = np.dtype(
    [
        ("time", "<u4"),
        ("ax", "<f4"),
        ("ay", "<f4"),
        ("az", "<f4"),
        ("gx", "<f4"),
        ("gy", "<f4"),
        ("gz", "<f4"),
    ]
)

mag_dt = np.dtype(
    [
        ("time", "<u4"),
        ("mx", "<f4"),
        ("my", "<f4"),
        ("mz", "<f4"),
    ]
)

env_dt = np.dtype(
    [
        ("time", "<u4"),
        ("T", "<f4"),
        ("P", "<f4"),
    ]
)

ted_dt = np.dtype(
    [
        ("time", "<u4"),
        ("T_adc", "<i4"),
        ("T_V", "<f4"),
        ("T_R", "<f4"),
        ("I_mA", "<f4"),
        ("V_mV", "<f4"),
        ("P_mW", "<f4"),
    ]
)

audio_dt = np.dtype(
    [
        ("left", "<i2"),
        ("right", "<i2"),
    ]
)


class DataProcessor:
    HeaderSize = 16
    DefaultSyncword = b"\xD6\xBE\x89\x8E"

    def __init__(self, buffer_size: int, syncword: bytes = DefaultSyncword):
        self.crc = crc.Calculator(crc.Crc16.KERMIT, optimized=True)
        self.buffer = RingBuffer(buffer_size)
        self.syncword = syncword
        self.timestamp_offset_ms = 0

        # Metrics.
        self.invalid_header_crc = 0
        self.invalid_data_crc = 0
        self.packet_parse_errors = 0
        self.skipped_bytes = 0

    def increment_skip_count(self, n: int) -> None:
        self.skipped_bytes += n

    def skip(self, n: int) -> None:
        self.increment_skip_count(self.buffer.consume(n))

    def skip_to_syncword(self) -> bool:
        swi = self.buffer.find(self.syncword)
        if swi < 0:
            self.skip(self.buffer.used)
            return False
        if swi > 0:
            self.skip(swi)
        return True

    def peek_header(self) -> pd.DataFrame:
        if self.buffer.used < self.HeaderSize:
            return None

        header_data = self.buffer.peek(self.HeaderSize).tobytes()
        header = pd.DataFrame(
            np.frombuffer(
                header_data,
                dtype=header_dt,
            )
        )

        # Validate header CRC.
        if not self.crc.verify(header_data[:-2], header.header_crc[0]):
            # Invalid packet header. Skip one byte and skip to next syncword.
            self.skip(1)
            self.skip_to_syncword()
            raise InvalidHeaderChecksum()

        return header

    def _extract_dt_array(self, header, payload, dt):
        extracted = pd.DataFrame(np.frombuffer(payload, dtype=dt))
        if "time" in extracted.columns:
            extracted.time -= self.timestamp_offset_ms
            extracted.time = extracted.time * 1.0e-3
        return extracted

    def _extract_imu(self, header, payload):
        return self._extract_dt_array(header, payload, imu_dt)

    def _extract_mag(self, header, payload):
        return self._extract_dt_array(header, payload, mag_dt)

    def _extract_env(self, header, payload):
        return self._extract_dt_array(header, payload, env_dt)

    def _extract_ted(self, header, payload):
        return self._extract_dt_array(header, payload, ted_dt)

    def _extract_audio(self, header, payload):
        audio_data = self._extract_dt_array(header, payload, audio_dt)
        end_time = header.timestamp_ms[0] * 1.0e-3
        samples = len(audio_data)
        start_time = end_time - samples * (1.0 / 16000)
        audio_data["time"] = np.linspace(start_time, end_time, samples, endpoint=False)
        return audio_data

    def process(self, data: bytes) -> List[Tuple[PacketType, pd.DataFrame]]:
        # Add new data and skip to the next syncword.
        self.buffer.add(data)

        packets = []
        while True:
            # Look for next syncword.
            if not self.skip_to_syncword():
                break

            # Enough data for the header?
            if self.buffer.used < self.HeaderSize:
                break

            # Attempt to peek a packet header.
            try:
                header = self.peek_header()
            except InvalidHeaderChecksum:
                self.invalid_header_crc += 1
                continue
            if header is None:
                continue

            # Is there enough data in the buffer to read the entire packet?
            data_size = header.data_size[0]
            if self.buffer.used < self.HeaderSize + data_size:
                break

            # Consume the header.
            self.buffer.consume(self.HeaderSize)

            # Check payload CRC.
            payload = self.buffer.peek(data_size).tobytes()
            if not self.crc.verify(payload, header.data_crc[0]):
                self.invalid_data_crc += 1
                self.increment_skip_count(self.HeaderSize)
                continue

            # Process the packet.
            try:
                packet_type = PacketType(header.data_type[0])

                if packet_type == PacketType.IMU:
                    packets.append((PacketType.IMU, self._extract_imu(header, payload)))
                elif packet_type == PacketType.Mag:
                    packets.append((PacketType.Mag, self._extract_mag(header, payload)))
                elif packet_type == PacketType.Env:
                    packets.append((PacketType.Env, self._extract_env(header, payload)))
                elif packet_type == PacketType.TED:
                    packets.append((PacketType.TED, self._extract_ted(header, payload)))
                elif packet_type == PacketType.Audio:
                    packets.append(
                        (PacketType.Audio, self._extract_audio(header, payload))
                    )
                else:
                    # This shouldn't happen. PacketType(...) should raise a ValueError.
                    raise UnknownPacketType()

            except (ValueError, UnknownPacketType) as e:
                if isinstance(e, UnknownPacketType):
                    print(
                        f"Unknown packet type {header.data_type[0]:%02X} at index {idx}"
                    )
                elif isinstance(e, ValueError):
                    print(f"ValueError at {idx}:")
                    traceback.print_exc()
                print("Header:", header)

                self.skip(1)
                self.packet_parse_errors += 1
                self.increment_skip_count(self.HeaderSize)
                continue

            # Consume the data.
            self.buffer.consume(data_size)

        return packets


def process_data(data: bytes, ring_buffer_size=None, chunk_size=None, metrics=False):
    if ring_buffer_size is None:
        ring_buffer_size = len(data)
    if chunk_size is None:
        chunk_size = ring_buffer_size
    processor = DataProcessor(ring_buffer_size)

    t0 = time.time()

    si = 0
    packets = []
    while si < len(data):
        ei = si + chunk_size
        if ei > len(data):
            ei = len(data)
        packets.extend(processor.process(data[si:ei]))
        si = ei

    imu = []
    mag = []
    env = []
    ted = []
    audio = []

    if packets is not None:
        for packet_type, packet_data in packets:
            if packet_type == PacketType.Audio:
                audio.append(packet_data)
            elif packet_type == PacketType.TED:
                ted.append(packet_data)
            elif packet_type == PacketType.IMU:
                imu.append(packet_data)
            elif packet_type == PacketType.Env:
                env.append(packet_data)
            elif packet_type == PacketType.Mag:
                mag.append(packet_data)
            else:
                print(f"Unknown packet type: {packet_type}")

    if len(imu) == 0:
        print("No IMU data found")
        imu_df = None
    else:
        imu_df = pd.concat(imu, ignore_index=True)

    if len(mag) == 0:
        print("No magnetometer data found")
        mag_df = None
    else:
        mag_df = pd.concat(mag, ignore_index=True)

    if len(env) == 0:
        print("No environmental data found")
        env_df = None
    else:
        env_df = pd.concat(env, ignore_index=True)

    if len(ted) == 0:
        print("No TED data found")
        ted_df = None
    else:
        ted_df = pd.concat(ted, ignore_index=True)

    if len(audio) == 0:
        print("No audio data found")
        audio_df = None
    else:
        audio_df = pd.concat(audio, ignore_index=True)

    t1 = time.time()

    if metrics:
        dt = t1 - t0
        data_dt = (
            max(audio_df.time) - min(audio_df.time)
            if (audio_df is not None) and (len(audio_df) > 0)
            else -1
        )

        print(
            f"Dropped {processor.skipped_bytes} of {len(data)} bytes "
            f"({processor.skipped_bytes*100.0/len(data):.2f}%)"
        )
        print(f"Invalid header CRCs: {processor.invalid_header_crc}")
        print(f"Invalid data CRCs: {processor.invalid_data_crc}")
        print(
            f"Processed {len(data)/1024.0/1024.0:.2f} MB data "
            f"({data_dt:.2f} s) in {dt:.2f} seconds"
        )

    return imu_df, mag_df, env_df, ted_df, audio_df


def process_file(filename: str, *args, **kwargs):
    data = open(filename, "rb").read()
    return process_data(data, *args, **kwargs)


def app():
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract a FricSense binary data file."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input binary file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory. Leave empty to extract to the same directory as the input file.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=None,
        help="Output file prefix. Leave empty to use the input file name.",
    )
    args = parser.parse_args()

    # Make sure input exists and is a file.
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        exit()
    if not os.path.isfile(args.input):
        print(f"Input {args.input} is not a file.")
        exit()

    # If an output directory was provided, make sure it exists and is a directory.
    if args.output is not None:
        if not os.path.exists(args.output):
            print(f"Output directory {args.output} does not exist.")
            exit()
        if not os.path.isdir(args.output):
            print(f"Output {args.output} is not a directory.")
            exit()

    # Otherwise, use the same directory as the input.
    else:
        args.output = os.path.dirname(os.path.realpath(args.input))

    # If no prefix was provided, use the input file name.
    if args.prefix is None:
        args.prefix = os.path.splitext(os.path.basename(args.input))[0]

    # Process the file.
    print(f"Processing {args.input}...")
    imu, mag, env, ted, audio = process_file(args.input, metrics=True)

    # Write the data to CSV files.
    print(f"Writing data to CSV files in {args.output}/{args.prefix}...")
    if imu is not None:
        imu.to_csv(f"{args.output}/{args.prefix}_imu.csv", index=False)
    if mag is not None:
        mag.to_csv(f"{args.output}/{args.prefix}_mag.csv", index=False)
    if env is not None:
        env.to_csv(f"{args.output}/{args.prefix}_env.csv", index=False)
    if ted is not None:
        ted.to_csv(f"{args.output}/{args.prefix}_ted.csv", index=False)
    if audio is not None:
        audio.to_csv(f"{args.output}/{args.prefix}_audio.csv", index=False)

    print("Done.")
