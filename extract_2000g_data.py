"""
Extract 2000g accelerometer data from enDAQ .IDE files

Processes all .IDE files in a specified folder and extracts data from 2000g accelerometer channels.
Exports each file's 2000g data to CSV.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import endaq.ide


def find_2000g_channels(doc):
    """
    Find channels corresponding to 2000g accelerometer.

    Args:
        doc: endaq.ide Dataset object

    Returns:
        List of channel IDs for 2000g accelerometer
    """
    # Get all channels
    channels = doc.channels

    # Look for channels with "2000" or similar in name/description
    # enDAQ sensors typically label high-g accelerometers clearly
    high_g_channels = []

    for channel_id, channel in channels.items():
        # Check channel name and sensor info
        channel_name = str(channel.name).lower() if hasattr(channel, 'name') else ''

        # Common patterns for 2000g accelerometers
        if '2000' in channel_name or '2000g' in channel_name:
            high_g_channels.append(channel_id)
            print(f"  Found 2000g channel: {channel.name} (ID: {channel_id})")

        # Alternative: check sensor object if available
        elif hasattr(channel, 'sensor') and channel.sensor:
            sensor_name = str(channel.sensor.name).lower()
            if '2000' in sensor_name or 'high' in sensor_name:
                high_g_channels.append(channel_id)
                print(f"  Found high-g channel: {channel.name} (ID: {channel_id})")

    return high_g_channels


def extract_2000g_data(ide_file_path, output_dir=None):
    """
    Extract 2000g accelerometer data from an IDE file.

    Args:
        ide_file_path: Path to .IDE file
        output_dir: Directory to save CSV output (default: same as IDE file)

    Returns:
        DataFrame with 2000g data, or None if no 2000g channels found
    """
    ide_path = Path(ide_file_path)

    if not ide_path.exists():
        print(f"Error: File not found: {ide_path}")
        return None

    print(f"\nProcessing: {ide_path.name}")

    try:
        # Open IDE file
        doc = endaq.ide.get_doc(str(ide_path))

        # Find 2000g channels
        high_g_channel_ids = find_2000g_channels(doc)

        if not high_g_channel_ids:
            print(f"  Warning: No 2000g channels found in {ide_path.name}")
            # List all available channels for debugging
            print("  Available channels:")
            for channel_id, channel in doc.channels.items():
                print(f"    - {channel.name} (ID: {channel_id})")
            return None

        # Convert to pandas DataFrame
        # Get only the 2000g channels
        df_full = endaq.ide.to_pandas(doc)

        # Filter columns that belong to 2000g channels
        # Column names typically include channel info
        high_g_columns = []
        for col in df_full.columns:
            # Check if column belongs to any of our high-g channels
            for ch_id in high_g_channel_ids:
                if str(ch_id) in str(col) or '2000' in str(col).lower():
                    high_g_columns.append(col)
                    break

        if not high_g_columns:
            print(f"  Warning: Could not extract 2000g data columns")
            return None

        df_2000g = df_full[high_g_columns]

        print(f"  Extracted {len(df_2000g.columns)} channels with {len(df_2000g)} samples")
        print(f"  Columns: {list(df_2000g.columns)}")

        # Save to CSV
        if output_dir is None:
            output_dir = ide_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{ide_path.stem}_2000g.csv"
        df_2000g.to_csv(output_file)
        print(f"  Saved to: {output_file}")

        return df_2000g

    except Exception as e:
        print(f"  Error processing {ide_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_folder(folder_path, output_dir=None):
    """
    Process all .IDE files in a folder.

    Args:
        folder_path: Path to folder containing .IDE files
        output_dir: Directory to save CSV outputs (default: same as IDE files)
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return

    # Find all .IDE files
    ide_files = list(folder.glob('*.IDE')) + list(folder.glob('*.ide'))

    if not ide_files:
        print(f"No .IDE files found in {folder}")
        return

    print(f"Found {len(ide_files)} .IDE files")
    print("=" * 60)

    successful = 0
    failed = 0

    for ide_file in sorted(ide_files):
        df = extract_2000g_data(ide_file, output_dir)
        if df is not None:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed/No 2000g data: {failed}")


if __name__ == "__main__":
    # Usage:
    # python extract_2000g_data.py <path_to_Endaq_Readings_folder> [output_folder]

    if len(sys.argv) < 2:
        print("Usage: python extract_2000g_data.py <IDE_folder_path> [output_folder]")
        print("\nExample:")
        print("  python extract_2000g_data.py ./Endaq_Readings")
        print("  python extract_2000g_data.py ./Endaq_Readings ./output_2000g")
        sys.exit(1)

    ide_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    process_folder(ide_folder, output_folder)
