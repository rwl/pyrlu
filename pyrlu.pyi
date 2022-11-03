from numpy import int32, float64, complex128
from numpy.typing import NDArray


def factor_solve(n: int, rowind: NDArray[int32], colptr: NDArray[int32], nz: NDArray[float64], b: NDArray[float64],
                 trans: bool = False, par: bool = False): ...


def z_factor_solve(n: int, rowind: NDArray[int32], colptr: NDArray[int32], nz: NDArray[complex128],
                   b: NDArray[complex128], trans: bool = False, par: bool = False): ...
