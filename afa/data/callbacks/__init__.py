"""AFA callbacks related to data."""
from .dumper import ImageDumperCallback
from .evaluator import MeanIoUCallback

__all__ = ["MeanIoUCallback", "ImageDumperCallback"]
