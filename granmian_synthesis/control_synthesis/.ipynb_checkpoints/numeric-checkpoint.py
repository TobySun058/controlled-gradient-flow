from typing import Callable, Union, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array

try:
    import quadax

    QUADAX_IMPORTED = True
except:
    QUADAX_IMPORTED = False

SampledIntegrator: TypeAlias = Callable[[Array, Array, int], Array]
LinearSolver: TypeAlias = Callable[[Array, Array, float], Array]


def make_sampled_integrator(method: Union[str, SampledIntegrator] = "simpsons") -> SampledIntegrator:
    """
    Return a method for sampled integration. Depend on the configuration, it can use quadax or jax native functions.
    Parameters
    ----------
    method : str
        The integration method to use. Can be one of:
        - "simpsons": Use Simpson's rule.
        - "trapezoid": Use the trapezoidal rule.
        - Callable: A custom integration function that takes (y: ArrayLike, x: ArrayLike, axis: int) as arguments.
    Returns
    -------
    callable
        A function that takes (y: ArrayLike, x: ArrayLike, axis: int) and returns the integrated result as a jax.Array.


    """
    if callable(method):
        return method

    if method == "simpsons":
        if QUADAX_IMPORTED:
            return simpsons_quadax
        else:
            return simpsons_jax
    elif method == "trapezoid":
        if QUADAX_IMPORTED:
            return trapezoid_quadax
        else:
            return trapezoid_jax
    else:
        raise KeyError(f"Unknown method: {method!r}") from None


def make_linear_solver(method: Union[LinearSolver, str]) -> LinearSolver:
    """
    Return a function that computes the inverse of A multiplied by y using the specified method.
    Parameters
    ----------
    method : str or callable
        The method to use for computing the inverse. Can be one of:
        - "cholesky": Use Cholesky decomposition.
        - "svd": Use Singular Value Decomposition (SVD). (Moore-Penrose pseudo-inverse, or pinv)
        - "hermitian": Use Hermitian pseudo-inverse. (pinv with hermitian=True, rcond=1e-15)
        - "common": Use common matrix inversion. (jnp.linalg.inv)
        Alternatively, a callable function that takes (A, y, eps) as arguments can be provided.
    Returns
    -------
    callable
        A function that takes (A, y, eps) and returns the solution x to (A + eps*I)x = y. (eps defaults to 0.0)
    """
    if callable(method):
        func = method
    elif method == "cholesky":
        func = solve_cholesky
    elif method == "svd":
        func = solve_svd
    elif method == "hermitian":
        func = solve_hermitian
    elif method in ["inverse", "inv"]:
        func = solve_via_inv
    elif method in ["ls", "lstsq", "least_squares"]:
        func = solve_ls
    else:
        raise KeyError(f"Unknown inverse solver method: {method!r}") from None

    def inv(A: jax.Array, y: jax.Array, eps: float = 0.0) -> jax.Array:

        return func(A + eps * jnp.eye(A.shape[0]), y)

    return inv


def arccos_degree(x):
    return jnp.rad2deg(jnp.arccos(x))


def make_basis_function(
        m: int,
        factor: float,
        offset: float,
        basis_type: str, ) -> Callable[[jax.Array], jax.Array]:
    match basis_type:
        case "fourier":
            assert m % 2 == 1, "For fourier basis, m must be odd."
            return get_fourier_basis_func(m // 2, factor, offset)
        case "polynomial":
            assert m >= 2, "For polynomial basis, m must be at least 2."
            return get_polynomial_basis_func(m - 1, factor, offset)
        case "legendre":
            raise NotImplementedError("Legendre basis.")
        case _:
            raise KeyError(f"Unknown basis type: {basis_type!r}") from None


def get_fourier_basis_func(m, factor, offset):
    coeff = jnp.arange(1, m + 1)
    scale0 = 1.0 / jnp.sqrt(factor)
    scale = jnp.sqrt(2.0 / factor)

    def basis(t):
        x = (t - offset) / factor
        basis_0 = jnp.array([scale0])
        basis_sin = scale * jnp.sin(jnp.pi * coeff * x)
        basis_cos = scale * jnp.cos(jnp.pi * coeff * x)
        return jnp.concatenate((basis_0, basis_sin, basis_cos)).reshape(2 * m + 1, 1)

    return basis


def get_polynomial_basis_func(m, factor, offset):
    n = jnp.arange(0, m + 1)
    coeff = jnp.sqrt((2.0 * n + 1.0) / (factor ** (2.0 * n + 1.0)))

    def basis(t):
        t = jnp.atleast_1d(t)  # (N,)
        x = (t - offset)[:, None]  # (N,1)
        return (coeff[None, :] * (x ** n)).reshape(t.shape[0], m + 1, 1)

    return basis


def simps(y, x=None, dx=1, axis=-1):
    """
    Modified from scipy.integrate.simps by replacing numpy with jax.numpy
    source: https://github.com/scipy/scipy/blob/v0.14.0/scipy/integrate/quadrature.py
    """

    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    def _basic_simps(y, start, stop, x, dx, axis):
        nd = len(y.shape)
        if start is None:
            start = 0
        step = 2
        slice_all = (slice(None),) * nd
        slice0 = tupleset(slice_all, axis, slice(start, stop, step))
        slice1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        slice2 = tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

        if x is None:  # Even spaced Simpson's rule.
            result = jnp.sum(
                dx / 3.0 * (y[slice0] + 4 * y[slice1] + y[slice2]),
                axis=axis)
        else:
            # Account for possibly different spacings.
            #    Simpson's rule changes a bit.
            h = jnp.diff(x, axis=axis)
            sl0 = tupleset(slice_all, axis, slice(start, stop, step))
            sl1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
            h0 = h[sl0]
            h1 = h[sl1]
            hsum = h0 + h1
            hprod = h0 * h1
            h0divh1 = h0 / h1
            tmp = hsum / 6.0 * (y[slice0] * (2 - 1.0 / h0divh1) +
                                y[slice1] * hsum * hsum / hprod +
                                y[slice2] * (2 - h0divh1))
            result = jnp.sum(tmp, axis=axis)
        return result

    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError(
                "If given, shape of x must be 1-d or the "
                "same as y.")
        if x.shape[axis] != N:
            raise ValueError(
                "If given, length of x along axis must be the "
                "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice1 = (slice(None),) * nd
        slice2 = (slice(None),) * nd

        slice1 = tupleset(slice1, axis, -1)
        slice2 = tupleset(slice2, axis, -2)
        if x is not None:
            last_dx = x[slice1] - x[slice2]
        val += 0.5 * last_dx * (y[slice1] + y[slice2])
        result = _basic_simps(y, 0, N - 3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals

        slice1 = tupleset(slice1, axis, 0)
        slice2 = tupleset(slice2, axis, 1)
        if x is not None:
            first_dx = x[tuple(slice2)] - x[tuple(slice1)]
        val += 0.5 * first_dx * (y[slice2] + y[slice1])
        result += _basic_simps(y, 1, N - 2, x, dx, axis)

        val /= 2.0
        result /= 2.0
        result = result + val
    else:
        result = _basic_simps(y, 0, N - 2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


def simpsons_jax(y: ArrayLike, x: ArrayLike, axis: int = 0) -> Array:
    return simps(y=y, x=x, axis=axis)


def simpsons_quadax(y: ArrayLike, x: ArrayLike, axis: int = 0) -> Array:
    return quadax.simpson(y=y, x=x, axis=axis)


def trapezoid_jax(y: ArrayLike, x: ArrayLike, axis: int = 0) -> Array:
    return jax.scipy.integrate.trapezoid(y=y, x=x, axis=axis)


def trapezoid_quadax(y: ArrayLike, x: ArrayLike, axis: int = 0) -> Array:
    return quadax.trapezoid(y=y, x=x, axis=axis)


def solve_cholesky(A, y):
    L = jnp.linalg.cholesky(A)
    z = jnp.linalg.solve(L, y)
    x = jnp.linalg.solve(L.T, z)
    return x


def solve_ls(A, y):
    return jnp.linalg.lstsq(A, y)[0]


def solve_svd(A, y):
    return jnp.linalg.pinv(A) @ y


def solve_hermitian(A, y):
    return jnp.linalg.pinv(A, hermitian=True, rcond=1e-15) @ y  # , rtol=1e-15


def solve_via_inv(A, y):
    return jnp.linalg.inv(A) @ y
