"""
Explore what's actually in an enDAQ .IDE file
"""

import sys
from pathlib import Path
import endaq.ide
import numpy as np


def explore_ide_file(ide_file_path):
    """Explore all channels and data in an IDE file"""

    ide_path = Path(ide_file_path)
    print(f"Exploring: {ide_path.name}")
    print("=" * 80)

    # Open file
    doc = endaq.ide.get_doc(str(ide_path))

    # List all channels
    print(f"\nFound {len(doc.channels)} channels:")
    print("-" * 80)

    for channel_id, channel in doc.channels.items():
        channel_name = channel.name if hasattr(channel, 'name') else 'Unknown'
        print(f"\nChannel ID: {channel_id}")
        print(f"  Name: {channel_name}")

        if hasattr(channel, 'sensor') and channel.sensor:
            print(f"  Sensor: {channel.sensor.name if hasattr(channel.sensor, 'name') else 'Unknown'}")

        # Check for subchannels
        if hasattr(channel, 'subchannels'):
            subchannels = channel.subchannels
            if isinstance(subchannels, dict):
                subchannel_list = list(subchannels.values())
            else:
                subchannel_list = subchannels

            print(f"  Subchannels: {len(subchannel_list)}")

            for i, subchannel in enumerate(subchannel_list):
                subchannel_name = subchannel.name if hasattr(subchannel, 'name') else f'Subchannel {i}'
                print(f"    [{i}] {subchannel_name}")

                # Try to get data info
                try:
                    session = subchannel.getSession()

                    # Check different methods to get data size
                    if hasattr(session, 'arraySlice'):
                        data = session.arraySlice()
                        print(f"        Data shape: {data.shape}")
                        print(f"        Data type: {data.dtype}")
                        if len(data) > 0:
                            print(f"        First few values: {data[:min(5, len(data))]}")

                    if hasattr(session, 'arrayRange'):
                        range_info = session.arrayRange()
                        print(f"        Range: {range_info}")

                    if hasattr(subchannel, 'sampleRate'):
                        print(f"        Sample rate: {subchannel.sampleRate} Hz")

                except Exception as e:
                    print(f"        Error accessing data: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use first file in Endaq_Readings as default
        default_folder = Path(__file__).parent.parent / "Endaq_Readings"
        ide_files = list(default_folder.glob('*.IDE')) + list(default_folder.glob('*.ide'))
        if ide_files:
            explore_ide_file(ide_files[0])
        else:
            print("Usage: python explore_ide_file.py <path_to_IDE_file>")
            sys.exit(1)
    else:
        explore_ide_file(sys.argv[1])
