"""AFA data structures."""
from typing import Any, Callable, Dict, List

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]
NDArrayUI8 = npt.NDArray[np.uint8]

DictStrAny = Dict[str, Any]  # type: ignore
ListAny = List[Any]  # type: ignore
ArgsType = Any  # type: ignore
AugmentType = Callable[[Any], Any]  # type: ignore
