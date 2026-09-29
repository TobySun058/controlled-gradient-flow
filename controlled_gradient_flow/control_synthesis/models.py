from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, TypeAlias, override

import equinox as eqx
import jax
import jax.numpy as jnp
import scipy
from jaxtyping import Array

AffineDriftDynamics: TypeAlias = Callable[[float, Array], Array]
AffineControlDynamics: TypeAlias = Callable[[float, Array, Array], Array]


def load_mindy_model_v2(mat_path, beta=20.0 / 3.0, B=None):
    # 100D
    data = scipy.io.loadmat(mat_path)["allMdl"].flatten()
    N_models = data.shape[0]
    models = []
    if B is None:
        B = jnp.eye(100)
    for i in range(N_models):
        D = jnp.array(data[i][0, 0]["D"]).reshape(-1)
        W = jnp.array(data[i][0, 0]["W"])
        alpha = jnp.array(data[i][0, 0]["alpha"]).reshape(-1)
        _beta = beta * jnp.ones_like(alpha)
        state_dim = D.shape[0]
        C = jnp.zeros(state_dim)
        model = MINDyfMRI(D, W, B, C, alpha, _beta)
        models.append(model)
    return models

class AbstractControlAffineSystem(eqx.Module, ABC):
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError("Requires implementation of state_dim property")

    @abstractmethod
    def input_dim(self) -> int:
        raise NotImplementedError("Requires implementation of input_dim property")

    @abstractmethod
    def N_fn(self, t: float, x: Array) -> Array:
        raise NotImplementedError("Requires implementation of N_fn")

    @abstractmethod
    def B_fn(self, t: float, x: Array) -> Array:
        raise NotImplementedError("Requires implementation of B_fn")

    def F_fn(self, t: float, x: Array, u: Array) -> Array:
        return self.N_fn(t, x) + self.B_fn(t, x) @ u

    @cached_property
    def jacobian_N(self):
        return jax.jacfwd(self.N_fn, argnums=1)

    @cached_property
    def jacobian_F(self):
        return jax.jacfwd(self.F_fn, argnums=1)



class LinearSystem(AbstractControlAffineSystem):
    A: Array
    B: Array

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def input_dim(self) -> int:
        return self.B.shape[1]

    def N_fn(self, t: float, x: Array) -> Array:
        return self.A @ x

    def B_fn(self, t: float, x: Array) -> Array:
        return self.B

    @override
    def jacobian_N(self) -> Array:
        return self.A

    @override
    def jacobian_F(self) -> Array:
        return self.A


class HopfieldRNN(AbstractControlAffineSystem):
    D: Array
    W: Array
    B: Array
    C: Array
    psi: eqx.field(staic=True)

    @property
    def state_dim(self) -> int:
        return self.D.shape[0]

    @property
    def input_dim(self) -> int:
        return self.B.shape[1]

    def N_fn(self, t: float, x: Array) -> Array:
        return -self.D * x + self.W @ self.psi(x) + self.C

    def B_fn(self, t: float, x: Array) -> Array:
        return self.B


class MINDyfMRI(AbstractControlAffineSystem):
    D: Array
    W: Array
    B: Array
    C: Array
    Alpha: Array
    Beta: Array

    @property
    def state_dim(self) -> int:
        return self.D.shape[0]

    @property
    def input_dim(self) -> int:
        return self.B.shape[1]

    def N_fn(self, t: float, x: Array) -> Array:
        def psi(x, alpha, beta):
            s1 = jnp.sqrt(alpha ** 2 + (beta * x + 0.5) ** 2)
            s2 = jnp.sqrt(alpha ** 2 + (beta * x - 0.5) ** 2)
            return s1 - s2

        return -self.D * x + self.W @ psi(x, self.Alpha, self.Beta) + self.C

    def B_fn(self, t: float, x: Array) -> Array:
        return self.B


class Pendulum(AbstractControlAffineSystem):
    g: float
    l: float
    m: float
    lam: float

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def input_dim(self) -> int:
        return 1

    def N_fn(self, t: float, x: Array) -> Array:
        a = self.g / self.l
        b = 1 / (self.m * self.l ** 2)
        gamma = self.lam * b
        return jnp.array([x[1], -a * jnp.sin(x[0]) - gamma * x[1]]).reshape(-1)

    def B_fn(self, t: float, x: Array) -> Array:
        b = 1 / (self.m * self.l ** 2)
        return jnp.array([[0], [b]])


class Unicycle(AbstractControlAffineSystem):
    W: float

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def input_dim(self) -> int:
        return 2

    def N_fn(self, t: float, x: Array) -> Array:
        return jnp.array([0, 0, 0])

    def B_fn(self, t: float, x: Array) -> Array:
        return jnp.array([[jnp.cos(x[2]), 0], [jnp.sin(x[2]), 0], [0, self.W]])
