"""Metrics logging utilities."""

from collections import defaultdict
from typing import Dict, List, Any
import csv
import os
from datetime import datetime


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str = "data/logs"):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.current_episode = 0
        
        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"metrics_{timestamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None
        self.csv_initialized = False
        self.csv_fieldnames = ["step"]  # Track all fieldnames

    def log(self, key: str, value: float, step: int = None) -> None:
        """
        Log a metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step/episode number (uses current_episode if None)
        """
        if step is None:
            step = self.current_episode
        
        self.metrics[key].append((step, value))
        
        # Add field to fieldnames if not present
        if key not in self.csv_fieldnames:
            self.csv_fieldnames.append(key)
            # Recreate writer with new fieldnames
            if self.csv_initialized:
                # Need to recreate the file with new header
                self.csv_file.close()
                # Read existing data if any
                existing_data = []
                if os.path.exists(self.csv_path):
                    try:
                        with open(self.csv_path, "r", newline="") as f:
                            reader = csv.DictReader(f)
                            existing_data = list(reader)
                    except Exception:
                        existing_data = []
                
                # Reopen file and write with new header
                self.csv_file = open(self.csv_path, "w", newline="")
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
                # Write existing data back
                for row in existing_data:
                    # Fill missing fields with None
                    full_row = {field: row.get(field, None) for field in self.csv_fieldnames}
                    self.csv_writer.writerow(full_row)
                self.csv_file.flush()
            else:
                # First time initialization
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
                self.csv_initialized = True
        
        # Write row with all fields (missing ones will be None)
        # Use the last known values for all fields at this step
        row = {"step": step}
        for field in self.csv_fieldnames[1:]:  # Skip "step"
            if field == key:
                row[field] = value
            else:
                # Get last value for this field at this step or before
                if self.metrics.get(field):
                    # Find the last value for this field at or before this step
                    last_value = None
                    for s, v in reversed(self.metrics[field]):
                        if s <= step:
                            last_value = v
                            break
                    row[field] = last_value
                else:
                    row[field] = None
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def log_dict(self, metrics_dict: Dict[str, float], step: int = None) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric names to values
            step: Step/episode number
        """
        if step is None:
            step = self.current_episode
        
        # Add new fields to fieldnames if needed
        new_fields = [key for key in metrics_dict.keys() if key not in self.csv_fieldnames]
        if new_fields:
            self.csv_fieldnames.extend(new_fields)
            # Recreate writer with new fieldnames
            if self.csv_initialized:
                # Need to recreate the file with new header
                self.csv_file.close()
                # Read existing data if any
                existing_data = []
                if os.path.exists(self.csv_path):
                    with open(self.csv_path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        existing_data = list(reader)
                
                # Reopen file and write with new header
                self.csv_file = open(self.csv_path, "w", newline="")
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
                # Write existing data back
                for row in existing_data:
                    # Fill missing fields with None
                    full_row = {field: row.get(field, None) for field in self.csv_fieldnames}
                    self.csv_writer.writerow(full_row)
            else:
                # First time initialization
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
                self.csv_initialized = True
        
        # Build row with all fields
        row = {"step": step}
        for field in self.csv_fieldnames[1:]:  # Skip "step"
            if field in metrics_dict:
                row[field] = metrics_dict[field]
            else:
                # Get last value for this field, or None if not present
                if self.metrics.get(field):
                    row[field] = self.metrics[field][-1][1]  # Last value
                else:
                    row[field] = None
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        # Store in memory
        for key, value in metrics_dict.items():
            self.metrics[key].append((step, value))

    def increment_episode(self) -> None:
        """Increment current episode counter."""
        self.current_episode += 1

    def get_metric(self, key: str) -> List[tuple]:
        """
        Get all logged values for a metric.
        
        Args:
            key: Metric name
            
        Returns:
            List of (step, value) tuples
        """
        return self.metrics.get(key, [])

    def close(self) -> None:
        """Close the logger and CSV file."""
        if self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

