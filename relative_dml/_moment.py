"""Multiplicative moment extracted from the original manuscript experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

Array = NDArray[np.float64]


@dataclass
class Estimate:
    theta: list[float]
    se: list[float]
    moment_norm: float
    jac_condition: float


def fit_multiplicative_dml(a: Array, y: Array, v: Array,
                           b_hat: Array, m_a_hat: Array,
                           start: Array | None = None) -> Estimate:
    """Resolve o momento CQ-DML para g(a,x)=a v(x)'theta.

    b_hat e m_a_hat devem vir de folds externos ou ser funções pré-fixadas.
    O sandwich é justificado na interseção das nuisances sob taxas DML;
    não reivindica inferência DR sob misspecificação arbitrária de nuisance.
    """
    a, y = np.asarray(a, float), np.asarray(y, float)
    v = np.asarray(v, float)
    b_hat, m_a_hat = np.asarray(b_hat, float), np.asarray(m_a_hat, float)
    if any(z.ndim != 1 for z in (a, y, b_hat, m_a_hat)):
        raise ValueError('a, y, b_hat e m_a_hat devem ser vetores.')
    n = len(a)
    if v.ndim != 2 or not v.shape[1] or n <= v.shape[1] or any(len(z) != n for z in (y, v, b_hat, m_a_hat)):
        raise ValueError('Dimensões incompatíveis.')
    if not all(np.isfinite(z).all() for z in (a, y, v, b_hat, m_a_hat)):
        raise ValueError('Valores ausentes ou não finitos.')
    if not np.isin(y, [0, 1]).all() or not np.any(y == 1):
        raise ValueError('y deve ser binário e conter conversores.')
    if np.any((b_hat < 0) | (b_hat > 1)):
        raise ValueError('b_hat deve estar entre 0 e 1.')
    h = a[:, None]*v
    r = (a-m_a_hat)[:, None]*v

    def moment(theta):
        eta = -h@theta
        if np.max(np.abs(eta)) > 100:
            raise FloatingPointError('Extrapolação numérica na busca da raiz.')
        res = y*np.exp(eta)-b_hat
        return (r*res[:, None]).mean(axis=0)

    def jac(theta):
        wy = y*np.exp(-h@theta)
        return -(r.T@(h*wy[:, None]))/n

    solution = root(moment, np.zeros(v.shape[1]) if start is None else start,
                    jac=jac, method='hybr', options={'xtol': 1e-10})
    norm = float(np.linalg.norm(moment(solution.x)))
    if (not solution.success and norm > 1e-7) or norm > 1e-6:
        raise RuntimeError(f'Raiz não convergiu: {solution.message}; norma={norm}')
    j = jac(solution.x)
    condition = float(np.linalg.cond(j))
    if condition > 1e10:
        raise RuntimeError('Jacobiano mal condicionado: revisar suporte/modelo.')
    psi = r*(y*np.exp(-h@solution.x)-b_hat)[:, None]
    omega = (psi.T@psi)/n
    invj = np.linalg.inv(j)
    cov = invj@omega@invj.T/n
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return Estimate(solution.x.tolist(), se.tolist(), norm, condition)
