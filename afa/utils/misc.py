"""AFA miscellanous functions."""
from time import perf_counter
from typing import Optional, no_type_check

import torch
from packaging import version
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.seed import seed_everything


def is_torch_tf32_available() -> bool:
    """Check if torch TF32 is available."""
    return not (
        not torch.cuda.is_available()
        or torch.version.cuda is None
        or torch.cuda.get_device_properties(torch.cuda.current_device()).major
        < 8
        or int(torch.version.cuda.split(".", maxsplit=1)[0]) < 11
        or version.parse(torch.__version__) < version.parse("1.7")
    )


@rank_zero_only
def init_random_seed(seed: Optional[int]) -> None:
    """Initialize random seed."""
    seed_everything(seed, workers=True)


@no_type_check
def timeit(func):
    """Function to be used as decorator to time a function."""

    def timed(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        toc = perf_counter()
        print(f"{func.__name__}  {(toc - tic) * 1000:.2f} ms")
        return result

    return timed


class Timer:  # pragma: no cover
    """Timer class based on perf_counter."""

    def __init__(self) -> None:
        """Init."""
        self._tic = perf_counter()
        self._toc: Optional[float] = None
        self.paused = False

    def reset(self) -> None:
        """Reset timer."""
        self._tic = perf_counter()
        self._toc = None
        self.paused = False

    def pause(self) -> None:
        """Pause function."""
        if self.paused:
            raise ValueError("Timer already paused!")
        self._toc = perf_counter()
        self.paused = True

    def resume(self) -> None:
        """Resume function."""
        if self.paused:
            raise ValueError("Timer is not paused!")
        assert self._toc is not None
        self._tic = perf_counter() - (self._toc - self._tic)
        self._toc = None
        self.paused = False

    def time(self, milliseconds: bool = False) -> float:
        """Return elapsed time."""
        if not self.paused:
            self._toc = perf_counter()
        assert self._toc is not None
        time_elapsed = self._toc - self._tic
        if milliseconds:
            return time_elapsed * 1000
        return time_elapsed
