from enum import Enum
from typing import ClassVar


class AggregationStrategy(Enum):
    """This identifies the strategy used to deal with overlapping probabilities."""
    average_probabilities = 0
