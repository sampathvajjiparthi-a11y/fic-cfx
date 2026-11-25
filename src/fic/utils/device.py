import torch


def get_device(choice: str = "auto") -> torch.device:
    """Pick cuda (NVIDIA GPU), mps (Apple Silicon) or cpu."""
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
