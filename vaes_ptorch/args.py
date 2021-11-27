from dataclasses import dataclass


@dataclass
class TrainArgs:
    num_epochs: int
    batch_size: int
    print_every: int = 0  # never print if zero
    smoothing: float = 0.9
