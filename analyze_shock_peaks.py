"""
Analyze shock peaks from enDAQ .IDE files

Extracts 2000g accelerometer data, calculates magnitude, finds peaks,
and creates visualizations comparing all recordings.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import endaq.ide
from datetime import datetime


def find_2000g_channels(doc):
    """
    Find channels corresponding to 2000g accelerometer.

    Returns:
        List of channel IDs for 2000g accelerometer
    """
    channels = doc.channels
    high_g_channels = []

    for channel_id, channel in channels.items():
        channel_name = str(channel.name).lower() if hasattr(channel, 'name') else ''

        if '2000' in channel_name or '2000g' in channel_name:
            high_g_channels.append(channel_id)
        elif hasattr(channel, 'sensor') and channel.sensor:
            sensor_name = str(channel.sensor.name).lower()
            if '2000' in sensor_name or 'high' in sensor_name:
                high_g_channels.append(channel_id)

    return high_g_channels


def extract_acceleration_magnitude(ide_file_path):
    """
    Extract 2000g accelerometer data and calculate magnitude.

    Args:
        ide_file_path: Path to .IDE file

    Returns:
        tuple: (DataFrame with magnitude, peak_value, peak_time) or (None, None, None) on error
    """
    ide_path = Path(ide_file_path)

    if not ide_path.exists():
        print(f"Error: File not found: {ide_path}")
        return None, None, None

    print(f"Processing: {ide_path.name}")

    try:
        # Open IDE file
        doc = endaq.ide.get_doc(str(ide_path))

        # Find 2000g channels - look through all channels
        accel_data = {'X': None, 'Y': None, 'Z': None}
        time_data = None

        for channel_id, channel in doc.channels.items():
            channel_name = str(channel.name).lower() if hasattr(channel, 'name') else ''

            # Check if this is a 2000g accelerometer channel
            if '2000' not in channel_name:
                continue

            print(f"  Found 2000g channel: {channel.name}")

            # Get subchannels (X, Y, Z axes)
            if hasattr(channel, 'subchannels'):
                subchannels = channel.subchannels
                # Handle both list and dict formats
                if isinstance(subchannels, dict):
                    subchannel_list = list(subchannels.values())
                else:
                    subchannel_list = subchannels

                for subchannel in subchannel_list:
                    subchannel_name = str(subchannel.name).upper() if hasattr(subchannel, 'name') else ''

                    # Extract data from subchannel
                    try:
                        # Get the data array
                        event_array = subchannel.getSession()
                        data = event_array.arraySlice()

                        # Data format: shape is (2, N) where row 0 is time, row 1 is values
                        # NOT (N, 2)! The enDAQ format has time in first row, data in second row
                        if len(data.shape) == 2:
                            if time_data is None:
                                time_data = data[0, :]  # First ROW is time
                            values = data[1, :]  # Second ROW is values
                        else:
                            values = data
                            if time_data is None:
                                # Generate time array based on sample rate
                                sample_rate = subchannel.sampleRate if hasattr(subchannel, 'sampleRate') else 1000
                                time_data = np.arange(len(values)) / sample_rate

                        # Determine which axis (X, Y, or Z)
                        if 'X' in subchannel_name or subchannel_name.endswith(' 0'):
                            accel_data['X'] = values
                            print(f"    X-axis: {len(values)} samples")
                        elif 'Y' in subchannel_name or subchannel_name.endswith(' 1'):
                            accel_data['Y'] = values
                            print(f"    Y-axis: {len(values)} samples")
                        elif 'Z' in subchannel_name or subchannel_name.endswith(' 2'):
                            accel_data['Z'] = values
                            print(f"    Z-axis: {len(values)} samples")

                    except Exception as e:
                        print(f"    Error extracting subchannel {subchannel_name}: {e}")
                        continue

        # Check if we got all three axes
        if accel_data['X'] is None or accel_data['Y'] is None or accel_data['Z'] is None or time_data is None:
            print(f"  Warning: Could not extract all axes")
            print(f"  Got: X={accel_data['X'] is not None}, Y={accel_data['Y'] is not None}, Z={accel_data['Z'] is not None}")
            return None, None, None

        # Create DataFrame
        df = pd.DataFrame({
            'time': time_data,
            'accel_x': accel_data['X'],
            'accel_y': accel_data['Y'],
            'accel_z': accel_data['Z']
        })

        # Calculate magnitude: √(x² + y² + z²)
        df['magnitude'] = np.sqrt(
            df['accel_x']**2 +
            df['accel_y']**2 +
            df['accel_z']**2
        )

        # Find peak
        peak_idx = df['magnitude'].idxmax()
        peak_value = df.loc[peak_idx, 'magnitude']
        peak_time = df.loc[peak_idx, 'time']

        print(f"  Peak: {peak_value:.1f} g at time {peak_time:.4f}s")
        print(f"  Samples: {len(df)}")

        return df, peak_value, peak_time

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def plot_individual_shock(df, filename, peak_value, peak_time, output_dir):
    """
    Create a plot for an individual shock recording.

    Args:
        df: DataFrame with time and magnitude columns
        filename: Original .IDE filename
        peak_value: Peak magnitude value
        peak_time: Time of peak
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot magnitude over time
    ax.plot(df['time'], df['magnitude'], 'b-', linewidth=0.5, label='Magnitude')

    # Mark peak
    ax.plot(peak_time, peak_value, 'r*', markersize=15, label=f'Peak: {peak_value:.1f} g')

    # Annotate peak
    ax.annotate(f'{peak_value:.1f} g',
                xy=(peak_time, peak_value),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Acceleration Magnitude (g)', fontsize=12)
    ax.set_title(f'Shock Acceleration: {filename}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save plot
    output_path = output_dir / f"{Path(filename).stem}_magnitude.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved plot: {output_path}")


def plot_peak_comparison(results, output_dir):
    """
    Create comparison plot of all peak values.

    Args:
        results: List of tuples (filename, peak_value, peak_time)
        output_dir: Directory to save plot
    """
    if not results:
        print("No results to plot")
        return

    # Sort by peak value (descending)
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    filenames = [Path(r[0]).stem for r in results_sorted]
    peak_values = [r[1] for r in results_sorted]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(range(len(filenames)), peak_values, color='steelblue', edgecolor='black', linewidth=1.2)

    # Color the highest peak differently
    if bars:
        bars[0].set_color('crimson')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, peak_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}g',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Test File', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Acceleration (g)', fontsize=12, fontweight='bold')
    ax.set_title('Shock Test Peak Accelerations Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(filenames)))
    ax.set_xticklabels(filenames, rotation=45, ha='right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # Add summary statistics
    mean_peak = np.mean(peak_values)
    std_peak = np.std(peak_values)
    max_peak = np.max(peak_values)
    min_peak = np.min(peak_values)

    stats_text = f'Max: {max_peak:.1f}g | Min: {min_peak:.1f}g | Mean: {mean_peak:.1f}g | Std: {std_peak:.1f}g'
    ax.text(0.5, 0.98, stats_text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot
    output_path = output_dir / "peak_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison plot: {output_path}")

    # Also create a summary CSV
    summary_df = pd.DataFrame(results_sorted, columns=['Filename', 'Peak_g', 'Peak_Time'])
    csv_path = output_dir / "peak_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    plt.show()


def process_all_shocks(folder_path, output_dir=None):
    """
    Process all .IDE files in folder and create visualizations.

    Args:
        folder_path: Path to Endaq_Readings folder
        output_dir: Directory for output plots (default: folder_path/analysis)
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return

    # Create output directory
    if output_dir is None:
        output_dir = folder / "analysis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .IDE files
    ide_files = list(folder.glob('*.IDE')) + list(folder.glob('*.ide'))

    if not ide_files:
        print(f"No .IDE files found in {folder}")
        return

    print(f"Found {len(ide_files)} .IDE files")
    print("=" * 70)

    results = []  # Store (filename, peak_value, peak_time) tuples

    for ide_file in sorted(ide_files):
        df, peak_value, peak_time = extract_acceleration_magnitude(ide_file)

        if df is not None:
            # Plot individual shock
            plot_individual_shock(df, ide_file.name, peak_value, peak_time, output_dir)

            # Store result for comparison
            results.append((ide_file.name, peak_value, peak_time))

        print()

    print("=" * 70)
    print(f"Successfully processed {len(results)} files")

    # Create comparison plot
    if results:
        plot_peak_comparison(results, output_dir)

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        peak_values = [r[1] for r in results]
        print(f"Maximum peak:  {np.max(peak_values):.1f} g")
        print(f"Minimum peak:  {np.min(peak_values):.1f} g")
        print(f"Mean peak:     {np.mean(peak_values):.1f} g")
        print(f"Std deviation: {np.std(peak_values):.1f} g")
        print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Default to Endaq_Readings in parent directory
    default_folder = Path(__file__).parent.parent / "Endaq_Readings"

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        folder_path = default_folder
        output_path = None

    print(f"Processing folder: {folder_path}")
    print(f"Output directory: {output_path if output_path else folder_path / 'analysis'}")
    print()

    process_all_shocks(folder_path, output_path)
