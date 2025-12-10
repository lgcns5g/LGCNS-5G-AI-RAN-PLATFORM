#!/usr/bin/env python3  # noqa: EXE001
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.3.2",
#     "matplotlib>=3.5.0",
# ]
# ///

"""
Nsight Systems Statistics Analyzer.

Extracts and analyzes function execution statistics from Nsight Systems report files.
Converts .nsys-rep files to SQLite format for efficient processing and generates
statistical analysis with CCDF plots for function execution times.

Supports:
- Automatic conversion of .nsys-rep files to SQLite format
- NVTX event extraction and analysis
- Statistical analysis (percentiles, mean, std dev)
- Linear and logarithmic CCDF plot generation

Usage:
    # Direct execution:
    python nsys_stats.py report.nsys-rep function1 [function2 ...]

    # After pip install:
    nsys-stats report.nsys-rep function1 [function2 ...] [options]
"""

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

# Force non-interactive backend before importing matplotlib
# Prevents conflicts when called from Jupyter notebooks
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np


class WarningErrorHandler(logging.Handler):
    """Custom logging handler to track WARNING and ERROR messages."""

    def __init__(self) -> None:
        super().__init__()
        self.has_warnings = False
        self.has_errors = False

    def emit(self, record: logging.LogRecord) -> None:
        """Track WARNING and ERROR level messages."""
        if record.levelno >= logging.ERROR:
            self.has_errors = True
        elif record.levelno >= logging.WARNING:
            self.has_warnings = True

    def has_warning_or_error(self) -> bool:
        """Return True if any WARNING or ERROR messages were logged."""
        return self.has_warnings or self.has_errors


class NsysStatsAnalyzer:
    """Analyzer for NVTX function execution statistics from Nsight Systems files.

    Converts .nsys-rep files to SQLite format and analyzes NVTX events to generate
    comprehensive statistics and distribution plots.
    """

    def __init__(self, nsys_file: Path, function_names: list[str]) -> None:
        self.nsys_file = nsys_file
        self.function_names = function_names
        self.events: dict[str, list[dict[str, Any]]] = {name: [] for name in function_names}
        self.plot_count = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_execution_times_us(self, function_name: str | None = None) -> list[float]:
        """Get execution times in microseconds for a specific function or all functions.

        Returns all times if function_name is None, otherwise times for specific
        function.
        """
        if function_name:
            return [event["duration"] for event in self.events.get(function_name, [])]

        # Return all execution times for all functions
        all_times = []
        for events_list in self.events.values():
            all_times.extend(event["duration"] for event in events_list)
        return all_times

    def calculate_statistics(self, values: list[float]) -> dict[str, float]:
        """Calculate comprehensive statistics for a list of values.

        Args:
            values: List of numerical values to analyze

        Returns
        -------
            Dictionary containing statistical measures
        """
        if not values:
            return {}

        values_array = np.array(values)
        return {
            "count": len(values),
            "min": float(np.min(values_array)),
            "p10": float(np.percentile(values_array, 10)),
            "median": float(np.percentile(values_array, 50)),
            "mean": float(np.mean(values_array)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99)),
            "max": float(np.max(values_array)),
            "std": float(np.std(values_array)) if len(values) > 1 else 0.0,
        }

    def _log_statistics(self, stats: dict[str, float], title: str) -> None:
        """Log statistics in a consistent format.

        Args:
            stats: Statistics dictionary from calculate_statistics
            title: Title to display before statistics
        """
        self.logger.info(f"{title}:")
        self.logger.info(f"  Average: {stats['mean']:.3f} us")
        self.logger.info(f"  Median:  {stats['median']:.3f} us")
        self.logger.info(f"  10th:    {stats['p10']:.3f} us")
        self.logger.info(f"  95th:    {stats['p95']:.3f} us")
        self.logger.info(f"  99th:    {stats['p99']:.3f} us")
        self.logger.info(f"  Min:     {stats['min']:.3f} us")
        self.logger.info(f"  Max:     {stats['max']:.3f} us")
        self.logger.info(f"  Std:     {stats['std']:.3f} us")

    def _create_plot_statistics_and_markers(self, values_sorted: np.ndarray) -> list[tuple]:
        """Create statistics and markers for plotting.

        Args:
            values_sorted: Sorted array of values

        Returns
        -------
            List of (value, label, color) tuples for plotting markers
        """
        # Calculate basic statistics
        min_val = float(np.min(values_sorted))
        max_val = float(np.max(values_sorted))
        median_val = float(np.median(values_sorted))
        mean_val = float(np.mean(values_sorted))

        statistics = [
            (min_val, "Min", "blue"),
            (median_val, "Median", "green"),
            (mean_val, "Mean", "red"),
            (max_val, "Max", "purple"),
        ]

        # Mark percentiles
        percentiles = [95, 99]
        percentile_colors = ["orange", "brown"]

        for p, color in zip(percentiles, percentile_colors, strict=False):
            value = float(np.percentile(values_sorted, p))
            statistics.append((value, f"{p}th %ile", color))

        return statistics

    def load_data(self, *, keep_sqlite: bool = False, force_conversion: bool = False) -> bool:
        """Load and process NVTX events from the nsys file.

        Args:
            keep_sqlite: Whether to keep the temporary SQLite file
            force_conversion: Force SQLite conversion even if direct access works

        Returns
        -------
            True if data loaded successfully, False otherwise
        """
        temp_sqlite = f"temp_{self.nsys_file.name}.sqlite"
        temp_path = Path(temp_sqlite)

        try:
            # Try direct SQLite access first (fastest)
            if not force_conversion and self._try_direct_sqlite_access():
                self.logger.debug("Direct SQLite access is possible, but implementation needed...")
                # For now, fall back to conversion method

            # Convert nsys file to SQLite
            if not self._convert_nsys_to_sqlite(str(self.nsys_file), temp_sqlite):
                return False

            # Process the NVTX events from SQLite
            self.events = self._process_nvtx_events_sqlite(temp_sqlite)

            # Check if we found events for any function
            total_events = sum(len(events_list) for events_list in self.events.values())
            if total_events == 0:
                self.logger.error(
                    f"No events found for any of the functions: {self.function_names}"
                )
                return False

            # Log results for each function
            for func_name, events_list in self.events.items():
                if events_list:
                    self.logger.info(f"Loaded {len(events_list)} events for function '{func_name}'")
                else:
                    self.logger.warning(f"No events found for function '{func_name}'")

            self.logger.info(f"Total events loaded: {total_events}")
            return True

        finally:
            # Clean up temporary file
            if not keep_sqlite and temp_path.exists():
                temp_path.unlink()
            elif keep_sqlite:
                self.logger.info(f"Kept temporary SQLite file: {temp_sqlite}")

    def plot_cdf(self, values: list[float], title: str, xlabel: str, output_file: str) -> None:
        """Plot Cumulative Distribution Function."""
        if not values:
            self.logger.warning(f"No values to plot for {title}")
            return

        values_sorted = np.sort(values)
        y_vals = np.arange(1, len(values_sorted) + 1) / len(values_sorted)

        plt.figure(figsize=(12, 7))
        plt.plot(values_sorted, y_vals, linewidth=2, label=f"CDF (n={len(values)})")
        plt.grid(visible=True, alpha=0.3)

        # Create statistics markers
        statistics = self._create_plot_statistics_and_markers(values_sorted)

        # Plot all markers and create legend
        for value, label, color in statistics:
            plt.axvline(
                x=value,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{label}: {value:.3f} µs",
            )

        plt.xlabel(xlabel)
        plt.ylabel("P(X ≤ x)")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.xlim(left=0)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.plot_count += 1
        self.logger.info(f"[{self.plot_count}] CDF plot saved to {output_file}")
        plt.close()

    def plot_ccdf(self, values: list[float], title: str, xlabel: str, output_file: str) -> None:
        """Plot Complementary Cumulative Distribution Function (1 - CDF)."""
        if not values:
            self.logger.warning(f"No values to plot for {title}")
            return

        values_sorted = np.sort(values)
        y_vals = 1 - (np.arange(1, len(values_sorted) + 1) / len(values_sorted))

        plt.figure(figsize=(12, 7))
        plt.semilogy(values_sorted, y_vals, linewidth=2, label=f"CCDF (n={len(values)})")
        plt.grid(visible=True, alpha=0.3)

        # Create statistics markers
        statistics = self._create_plot_statistics_and_markers(values_sorted)

        # Plot all markers and create legend
        for value, label, color in statistics:
            plt.axvline(
                x=value,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{label}: {value:.3f} µs",
            )

        plt.xlabel(xlabel)
        plt.ylabel("P(X > x) [log scale]")
        plt.title(title)
        plt.legend(loc="upper right")
        plt.xlim(left=0)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.plot_count += 1
        self.logger.info(f"[{self.plot_count}] CCDF plot saved to {output_file}")
        plt.close()

    def print_summary(self) -> None:
        """Print comprehensive summary statistics for all functions."""
        total_events = sum(len(events_list) for events_list in self.events.values())
        if total_events == 0:
            self.logger.warning("No events to analyze")
            return

        self.logger.info("====== NSYS FUNCTION STATISTICS ======")
        self.logger.info(f"====== Functions: {', '.join(self.function_names)} ======")
        self.logger.info(f"Total function calls: {total_events}")

        # Analyze each function separately
        for function_name, events_list in self.events.items():
            if not events_list:
                continue

            self.logger.info(f"\n--- Analysis for function: {function_name} ---")
            self.logger.info(f"Function calls: {len(events_list)}")

            # Calculate execution time statistics for this function
            execution_times = self.get_execution_times_us(function_name)
            exec_stats = self.calculate_statistics(execution_times)

            self._log_statistics(exec_stats, f"Execution Time Statistics for {function_name}")

            # Thread distribution for this function
            thread_distribution = {}
            for event in events_list:
                thread_id = event["thread"]
                thread_distribution[thread_id] = thread_distribution.get(thread_id, 0) + 1

            self.logger.info(f"Thread distribution for {function_name}:")
            for thread_id, count in thread_distribution.items():
                percentage = count / len(events_list) * 100
                self.logger.info(f"  Thread {thread_id}: {count} calls ({percentage:.1f}%)")

        # Overall statistics across all functions
        if len(self.function_names) > 1:
            self.logger.info("\n--- Overall Statistics Across All Functions ---")
            all_execution_times = self.get_execution_times_us()
            all_exec_stats = self.calculate_statistics(all_execution_times)
            self._log_statistics(all_exec_stats, "Overall Execution Time Statistics")

        self.logger.info("======================================")

    def generate_plots(self, output_dir: Path) -> None:
        """Generate comprehensive plots for all functions."""
        # Initialize progress tracking
        self.plot_count = 0
        self.logger.info("Generating plots...")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots for each function
        for function_name, events_list in self.events.items():
            if not events_list:
                self.logger.warning(f"No events to plot for function '{function_name}'")
                continue

            execution_times = self.get_execution_times_us(function_name)
            if not execution_times:
                continue

            # Generate CDF plot for this function
            cdf_title = (
                f"Function '{function_name}' Execution Times CDF ({len(execution_times)} samples)"
            )
            cdf_output = str(output_dir / f"{function_name}_execution_times_cdf.png")
            self.plot_cdf(execution_times, cdf_title, "Execution Time (microseconds)", cdf_output)

            # Generate CCDF plot for this function
            ccdf_title = (
                f"Function '{function_name}' Execution Times CCDF ({len(execution_times)} samples)"
            )
            ccdf_output = str(output_dir / f"{function_name}_execution_times_ccdf.png")
            self.plot_ccdf(
                execution_times,
                ccdf_title,
                "Execution Time (microseconds)",
                ccdf_output,
            )

        # Generate combined plots if we have multiple functions
        if len(self.function_names) > 1:
            all_execution_times = self.get_execution_times_us()
            if all_execution_times:
                # Combined CDF plot
                combined_cdf_title = (
                    f"All Functions Execution Times CDF ({len(all_execution_times)} samples total)"
                )
                combined_cdf_output = str(output_dir / "all_functions_execution_times_cdf.png")
                self.plot_cdf(
                    all_execution_times,
                    combined_cdf_title,
                    "Execution Time (microseconds)",
                    combined_cdf_output,
                )

                # Combined CCDF plot
                combined_ccdf_title = (
                    f"All Functions Execution Times CCDF ({len(all_execution_times)} samples total)"
                )
                combined_ccdf_output = str(output_dir / "all_functions_execution_times_ccdf.png")
                self.plot_ccdf(
                    all_execution_times,
                    combined_ccdf_title,
                    "Execution Time (microseconds)",
                    combined_ccdf_output,
                )

        self.logger.info(f"Completed! Generated {self.plot_count} plots.")

    def _convert_nsys_to_sqlite(self, nsys_file: str, sqlite_file: str) -> bool:
        """Convert Nsight Systems report file to SQLite format - much faster than JSON.

        Args:
            nsys_file: Path to the .nsys-rep report file
            sqlite_file: Path for the output SQLite file

        Returns
        -------
            True if conversion successful, False otherwise
        """
        self.logger.info(f"Converting {nsys_file} to SQLite format...")

        cmd = [
            "nsys",
            "export",
            "--force-overwrite",
            "true",
            "--type=sqlite",
            f"--output={sqlite_file}",
            nsys_file,
        ]
        result = subprocess.run(  # noqa: S603
            cmd, check=False, capture_output=True, text=True
        )

        if result.returncode != 0:
            self.logger.error(f"Error converting report file: {result.stderr}")
            return False

        self.logger.info(f"Successfully converted to {sqlite_file}")
        return True

    def _process_nvtx_events_sqlite(self, sqlite_file: str) -> dict[str, list[dict[str, Any]]]:
        """Process SQLite database to extract NVTX events - faster than JSON parsing.

        Args:
            sqlite_file: Path to the SQLite database file

        Returns
        -------
            Dictionary mapping function names to lists of event dictionaries
        """
        events_by_function: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.function_names
        }
        self.logger.info(f"Processing {sqlite_file} for functions: {self.function_names}")

        try:
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()

            # Query to get NVTX events that match our target functions
            # The exact table structure may vary by nsys version,
            # so we'll try different approaches
            query_attempts = [
                # Modern nsys versions
                """
                SELECT text, start, end, (end - start) as duration, globalTid
                FROM NVTX_EVENTS
                WHERE text LIKE ?
                """,
                # Alternative table name
                """
                SELECT text, start, end, (end - start) as duration, globalTid
                FROM nvtx_events
                WHERE text LIKE ?
                """,
                # Older versions might use different column names
                """
                SELECT Text, Timestamp, EndTimestamp,
                       (EndTimestamp - Timestamp) as Duration, GlobalTid
                FROM NVTX_EVENTS
                WHERE Text LIKE ?
                """,
            ]

            # Process each function name
            for function_name in self.function_names:
                search_pattern = f"%{function_name}%"
                results = None

                for query in query_attempts:
                    try:
                        cursor.execute(query, (search_pattern,))
                        results = cursor.fetchall()
                        if results:
                            self.logger.debug(
                                f"Successfully queried database for function '{function_name}'"
                            )
                            break
                    except sqlite3.OperationalError as e:
                        self.logger.debug(f"Query attempt failed: {e}")
                        continue

                if results is not None:
                    for row in results:
                        text, start_time, end_time, duration, thread_id = row

                        # Convert duration from nanoseconds to microseconds
                        duration_us = duration / 1000.0

                        event = {
                            "name": text,
                            "start": start_time,
                            "end": end_time,
                            "duration": duration_us,  # microseconds
                            "thread": thread_id or "Unknown",
                            "color": 0,  # SQLite export doesn't include color info
                        }
                        events_by_function[function_name].append(event)

            # Check if we need to debug table structure
            # (only if no results found for any function)
            if all(len(events) == 0 for events in events_by_function.values()):
                # If none of the standard queries work, check what tables are available
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                table_names = [table[0] for table in tables]
                self.logger.debug(f"Available tables: {table_names}")

                # Try to find NVTX-related tables
                for (table_name,) in tables:
                    if "nvtx" in table_name.lower():
                        try:
                            # Use parameterized query to avoid SQL injection
                            cursor.execute("PRAGMA table_info(?)", (table_name,))
                            columns = cursor.fetchall()
                            column_names = [col[1] for col in columns]
                            self.logger.debug(f"Table {table_name} columns: {column_names}")

                            # Try a generic query on this table with escaping
                            # Note: table names can't be parameterized, we validate them
                            # to prevent SQL injection by ensuring alphanumeric chars
                            if table_name.replace("_", "").replace(" ", "").isalnum():
                                query = f"SELECT * FROM {table_name} WHERE rowid <= 5"  # noqa: S608
                                cursor.execute(query)
                                sample = cursor.fetchall()
                                self.logger.debug(f"Sample data from {table_name}: {sample}")
                        except (sqlite3.Error, ValueError) as e:
                            self.logger.debug(f"Error examining table {table_name}: {e}")

            conn.close()

        except (sqlite3.Error, OSError):
            self.logger.exception("Error processing SQLite file")
            return events_by_function

        return events_by_function

    def _try_direct_sqlite_access(self) -> bool:
        """Try to access the internal SQLite database directly without conversion.

        Returns
        -------
            True if direct access is possible, False otherwise
        """
        self.logger.debug(f"Attempting direct SQLite access to {self.nsys_file}...")

        try:
            # Some nsys files are actually SQLite databases internally
            conn = sqlite3.connect(str(self.nsys_file))
            cursor = conn.cursor()

            # List available tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]

            # Look for NVTX tables
            nvtx_tables = [t for t in tables if "nvtx" in t.lower() or "NVTX" in t]

            if nvtx_tables:
                self.logger.debug(f"Found NVTX tables in direct access: {nvtx_tables}")

                for table in nvtx_tables:
                    # Validate table name to prevent injection
                    if not table.replace("_", "").replace(" ", "").isalnum():
                        continue

                    try:
                        # Try with the first function name for discovery
                        search_pattern = f"%{self.function_names[0]}%"
                        # Try common column name variations
                        text_cols = ["text", "Text", "name", "Name"]
                        start_cols = ["start", "Start", "timestamp", "Timestamp"]
                        end_cols = [
                            "end",
                            "End",
                            "endTimestamp",
                            "EndTimestamp",
                        ]

                        for text_col in text_cols:
                            for start_col in start_cols:
                                for end_col in end_cols:
                                    try:
                                        # Build query with validated column names
                                        # Table/column names validated above
                                        query = (
                                            f"SELECT {text_col}, {start_col}, "  # noqa: S608
                                            f"{end_col}, ({end_col} - {start_col}) "
                                            f"as duration FROM {table} WHERE "
                                            f"{text_col} LIKE ? LIMIT 5"
                                        )
                                        cursor.execute(query, (search_pattern,))
                                        results = cursor.fetchall()
                                        if results:
                                            self.logger.debug(
                                                f"Direct access successful! "
                                                f"Found {len(results)} sample events"
                                            )
                                            conn.close()
                                            return True
                                    except sqlite3.Error as e:
                                        self.logger.debug(f"Query failed: {e}")
                                        continue
                    except sqlite3.Error as e:
                        self.logger.debug(f"Table access failed: {e}")
                        continue

            conn.close()

        except (sqlite3.Error, OSError) as e:
            self.logger.debug(f"Direct SQLite access failed: {e}")

        return False


def main() -> None:
    """Analyze NVTX function statistics from Nsight Systems report files."""
    logger = logging.getLogger(__name__)

    # Track if any warnings or errors occurred
    warning_error_handler = WarningErrorHandler()

    parser = argparse.ArgumentParser(
        description="Analyze NVTX function statistics from Nsight Systems report files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Direct Python execution:
    python nsys_stats.py report.nsys-rep function1 function2
    python nsys_stats.py trace.nsys-rep my_kernel compute_sum --output-dir ./analysis

    # After pip install:
    nsys-stats report.nsys-rep function1 --verbose
    nsys-stats trace.nsys-rep my_kernel compute_sum orchestrate_computations \\
               --output-dir ./results
        """,
    )

    parser.add_argument(
        "nsys_file",
        type=Path,
        help="Path to .nsys-rep report file generated by Nsight Systems",
    )
    parser.add_argument(
        "function_names", nargs="+", help="Function names to analyze in NVTX events"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory for plot files (default: <input_filename>_analysis/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots, only print summary",
    )
    parser.add_argument("--keep-sqlite", action="store_true", help="Keep the temporary SQLite file")
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force SQLite conversion even if direct access works",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add the warning/error handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(warning_error_handler)

    if not args.nsys_file.exists():
        logger.error(f"File {args.nsys_file} does not exist")
        sys.exit(1)

    # Create analyzer and load data
    analyzer = NsysStatsAnalyzer(args.nsys_file, args.function_names)
    if not analyzer.load_data(keep_sqlite=args.keep_sqlite, force_conversion=args.force_conversion):
        sys.exit(1)

    # Print summary statistics
    analyzer.print_summary()

    # Generate plots if requested
    if not args.no_plots:
        output_dir = args.output_dir or (args.nsys_file.parent / f"{args.nsys_file.stem}_analysis")

        logger.info(f"Generating plots in directory: {output_dir}")
        analyzer.generate_plots(output_dir)
        logger.info(f"Analysis complete! Generated plots in directory: {output_dir}")
    else:
        logger.info("Analysis complete!")

    # Exit with error code if any warnings or errors occurred
    if warning_error_handler.has_warning_or_error():
        logger.error("Analysis completed with warnings or errors - exiting with error code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
