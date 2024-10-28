import serial
import time
import datetime as dt
import traceback

from . import find_fricsense


def size2str(size):
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.2f} MB"
    else:
        return f"{size / 1024 / 1024 / 1024:.2f} GB"


# Connect to serial port, send init command, then read data continuously and report throughput every second.
def read_serial(
    port: str,
    output_fn: str = None,
    postfix: str = None,
    buffer_size: int = 1024 * 1024,
):
    buffers = [
        bytearray(buffer_size),
        bytearray(buffer_size),
    ]
    which = 0
    buffer_i = 0

    # Connect to the serial port. Baud rate doesn't matter since it's actually USB.
    ser = serial.Serial(port, 9600)

    # Open the output file.
    if output_fn is None:
        output_fn = f"fricsense_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if postfix is not None:
        output_fn += f"_{postfix}"
    output_fn += ".bin"
    print(f"Opeining output file {output_fn}...")
    f = open(output_fn, "wb")

    # Send initialization command
    # ser.write(b"init_command")

    # Initialize variables
    start_time = time.time()
    total_bytes = 0
    iter_bytes = 0

    # Continuously read data and calculate throughput
    try:
        while True:
            # Read data from serial port
            data = ser.read(1000)

            # Calculate the number of bytes read
            num_bytes = len(data)

            # Store in the buffer.
            buffers[which][buffer_i : buffer_i + num_bytes] = data
            buffer_i += num_bytes

            # Update the total number of bytes
            total_bytes += num_bytes
            iter_bytes += num_bytes

            # Calculate the elapsed time
            elapsed_time = time.time() - start_time

            # Check if one second has passed. We will print the throughput and dump data to disk.
            if elapsed_time >= 3:
                # Calculate the throughput in bytes per second
                throughput = iter_bytes / elapsed_time / 1024.0

                # Print the throughput and write the buffer to the file.
                print(
                    f"Throughput: {throughput:.2f} KB/s (Writing {size2str(buffer_i)} to file)"
                )
                f.write(buffers[which][:buffer_i])

                # Switch the buffer.
                which = (which + 1) % 2
                buffer_i = 0

                # Reset the variables
                start_time = time.time()
                iter_bytes = 0

    except KeyboardInterrupt:
        print(f"Keyboard interrupt.")
        pass

    except serial.serialutil.SerialException:
        print(f"Serial port disconnect (probably).")
        traceback.print_exc()
        pass

    # Close the serial port and store the buffer to a file.
    print("Closing serial port.")
    ser.close()

    # Write anything remaining in the buffer.
    print(f"  -> Writing final {size2str(buffer_i)} to file...")
    f.write(buffers[which][:buffer_i])
    print(f"Total bytes written: {size2str(total_bytes)}")

    # Close the file.
    print("Closing output file.")
    f.close()

    print("Done.")


def app():
    import argparse

    parser = argparse.ArgumentParser(
        description="Read serial data from FricSense device connected over USB."
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default=None,
        help="Serial port to connect to. Leave empty to auto-detect.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file name. Leave empty to auto-generate.",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Postfix to append to the output file name after date/time.",
    )
    parser.add_argument(
        "-b",
        "--buffer-size",
        type=int,
        default=1024 * 1024,
        help="Size of the double buffer (bytes) to store data. Default is 1 MB.",
    )
    args = parser.parse_args()

    # If no port was provided, auto-detect.
    port = args.port
    if port is None:
        ports = find_fricsense()
        if len(ports) == 0:
            print("No FricSense device found.")
            exit()
        elif len(ports) > 1:
            print("Multiple FricSense devices found. Selecting the first one.")
        port = ports[0].device
        print(f"Connecting to {ports[0].manufacturer} - {ports[0].product} on {port}.")
    else:
        print(f"Connecting to {port}.")

    read_serial(port, args.output, args.postfix, args.buffer_size)
