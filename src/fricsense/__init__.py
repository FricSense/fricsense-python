def find_fricsense():
    from serial.tools.list_ports import comports

    return [port for port in comports() if port.product == "FricSense"]
