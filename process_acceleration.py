#!/usr/bin/env python3
"""
Process acceleration CSV file and add magnitude column.

Reads a CSV with columns: time, X acceleration, Y acceleration, Z acceleration
Adds a new column with the magnitude: sqrt(x^2 + y^2 + z^2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def process_acceleration_csv(input_file, output_file=None):
    """
    Process acceleration CSV and add magnitude column.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, defaults to input_file with _processed suffix)
    """
    # Read the CSV file - try with headers first
    print(f"Reading {input_file}...")

    # Try to detect if file has headers
    try:
        df_test = pd.read_csv(input_file, nrows=1)
        first_row = df_test.iloc[0].tolist()

        # If first row looks like numbers, assume no header
        if all(isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','').replace('-','').replace('e','').replace('E','').replace('+','').isdigit()) for x in first_row):
            print("No header detected. Reading with column names: time, accel_x, accel_y, accel_z")
            df = pd.read_csv(input_file, header=None, names=['time', 'accel_x', 'accel_y', 'accel_z'])
            time_col, x_col, y_col, z_col = 'time', 'accel_x', 'accel_y', 'accel_z'
        else:
            # Has headers
            df = pd.read_csv(input_file)
            print(f"Columns found: {list(df.columns)}")

            # Detect column names (handle different naming conventions)
            time_col = None
            x_col = None
            y_col = None
            z_col = None

            for col in df.columns:
                col_lower = col.lower().strip()
                if 'time' in col_lower:
                    time_col = col
                elif 'x' in col_lower:
                    x_col = col
                elif 'y' in col_lower:
                    y_col = col
                elif 'z' in col_lower:
                    z_col = col

            # If automatic detection failed, assume order: time, x, y, z
            if time_col is None or x_col is None or y_col is None or z_col is None:
                print("Auto-detection failed. Assuming column order: time, X, Y, Z")
                cols = df.columns.tolist()
                if len(cols) >= 4:
                    time_col, x_col, y_col, z_col = cols[0], cols[1], cols[2], cols[3]
                else:
                    raise ValueError(f"Expected at least 4 columns, found {len(cols)}")

    except Exception as e:
        print(f"Error reading file: {e}")
        raise

    print(f"Using columns: time={time_col}, X={x_col}, Y={y_col}, Z={z_col}")

    # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
    df['acceleration_magnitude'] = np.sqrt(
        df[x_col]**2 + df[y_col]**2 + df[z_col]**2
    )

    # Determine output filename
    if output_file is None:
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_processed.csv')
        else:
            output_file = input_file + '_processed.csv'

    # Save the result
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\nSummary:")
    print(f"Total rows: {len(df)}")
    print(f"Magnitude range: {df['acceleration_magnitude'].min():.4f} to {df['acceleration_magnitude'].max():.4f}")
    print(f"Magnitude mean: {df['acceleration_magnitude'].mean():.4f}")
    print(f"Magnitude std: {df['acceleration_magnitude'].std():.4f}")

    print(f"\nOutput saved to: {output_file}")

    # Create plots
    plot_acceleration_data(df, time_col, x_col, y_col, z_col, output_file)

    return df


def plot_acceleration_data(df, time_col, x_col, y_col, z_col, output_file):
    """
    Create plots of acceleration data.

    Args:
        df: DataFrame with acceleration data
        time_col: Name of time column
        x_col: Name of X acceleration column
        y_col: Name of Y acceleration column
        z_col: Name of Z acceleration column
        output_file: Base filename for saving plots
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Individual components
    axes[0].plot(df[time_col], df[x_col], label='X Acceleration', alpha=0.7, linewidth=0.8)
    axes[0].plot(df[time_col], df[y_col], label='Y Acceleration', alpha=0.7, linewidth=0.8)
    axes[0].plot(df[time_col], df[z_col], label='Z Acceleration', alpha=0.7, linewidth=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_title('Acceleration Components vs Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Magnitude
    axes[1].plot(df[time_col], df['acceleration_magnitude'],
                 label='Magnitude', color='red', linewidth=1.0)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration Magnitude (g)')
    axes[1].set_title('Acceleration Magnitude vs Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add peak annotation to magnitude plot
    max_idx = df['acceleration_magnitude'].idxmax()
    max_time = df[time_col].iloc[max_idx]
    max_mag = df['acceleration_magnitude'].iloc[max_idx]
    axes[1].plot(max_time, max_mag, 'ro', markersize=8)
    axes[1].annotate(f'Peak: {max_mag:.2f} g\nat t={max_time:.4f} s',
                     xy=(max_time, max_mag),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Save plot
    if output_file:
        plot_filename = output_file.replace('.csv', '.png')
    else:
        plot_filename = 'acceleration_plot.png'

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")

    # Show plot
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_acceleration.py <input_file.csv> [output_file.csv]")
        print("\nExample:")
        print("  python process_acceleration.py acceleration_data.csv")
        print("  python process_acceleration.py acceleration_data.csv output.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        process_acceleration_csv(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
