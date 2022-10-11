#! /usr/bin/env python3

import argparse
import re

from datetime import datetime

def extract_datetime(string: str, prefix: str, dt_format : str = '%d/%m/%Y %H:%M:%S'):
    needle = prefix + r'\s*(.+)'
    m = re.match(needle, string)  # assert start time location
    if not m:
        return
    dt_string = re.sub(r' +', ' ', m.group(1).strip())
    return datetime.strptime(dt_string, dt_format)

def main() -> None:
    ap = argparse.ArgumentParser(description="""
        Calculate runtime of libatoms program from start and stop time in its log file.
        Output in seconds.""")
    ap.add_argument('file', type=argparse.FileType('r'), help="log file to parse")
    args = ap.parse_args()

    init_prefix = 'libAtoms::Hello World:'
    final_prefix = 'libAtoms::Finalise:'

    init_lines = []
    final_lines = []
    for line in args.file:
        if line.startswith(init_prefix):
            init_lines.append(line)
        elif line.startswith(final_prefix):
            final_lines.append(line)

    if len(init_lines) < 1:
        quit(f"No init lines found. ({init_prefix})")
    if len(final_lines) < 1:
        quit(f"No final lines found. ({final_prefix})")

    init_dt = extract_datetime(init_lines[0], init_prefix)
    final_dt = extract_datetime(final_lines[-2], final_prefix)
    delta_dt = final_dt - init_dt
    print(int(delta_dt.total_seconds()))

if __name__ == '__main__':
    main()
