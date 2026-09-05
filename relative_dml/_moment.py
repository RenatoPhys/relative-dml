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
    # Covariance of theta_hat, including 1/n; defaults preserve old callers.
    cov: list[list[float]] | None = None
    jac_singular_values: list[float] | None = None


def fit_multiplicative_dml(t: Array, y: Array, v: Array,
                           b_hat: Array, m_t_hat: Array,
                           start: Array | None = None) -> Estimate:
    """Resolve o momento CQ-DML para g(t,x)=t v(x)'theta.

    b_hat e m_t_hat devem vir de folds externos ou ser funções pré-fixadas.
    O sandwich é justificado na interseção das nuisances sob taxas DML;
    não reivindica inferência DR sob misspecificação arbitrária de nuisance.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    v = np.asarray(v, float)
    b_hat, m_t_hat = np.asarray(b_hat, float), np.asarray(m_t_hat, float)
    if any(z.ndim != 1 for z in (t, y, b_hat, m_t_hat)):
        raise ValueError('t, y, b_hat e m_t_hat devem ser vetores.')
    n = len(t)
    if v.ndim != 2 or not v.shape[1] or n <= v.shape[1] or any(len(z) != n for z in (y, v, b_hat, m_t_hat)):
        raise ValueError('Dimensões incompatíveis.')
    if not all(np.isfinite(z).all() for z in (t, y, v, b_hat, m_t_hat)):
        raise ValueError('Valores ausentes ou não finitos.')
    return _fit_multiplicative_basis(t[:, None]*v, m_t_hat[:, None]*v,
                                     y, b_hat, start=start)


def _fit_multiplicative_basis(H: Array, m_H: Array, y: Array, b_hat: Array,
                              start: Array | None = None) -> Estimate:
    """Solve the multiplicative moment for a fixed, observation-aligned basis.

    m_H estimates E[H | X] out of fold. The iid sandwich includes 1/n and
    requires correct effect specification and the usual nuisance rates.
    """
    h, mh = np.asarray(H, float), np.asarray(m_H, float)
    y, b_hat = np.asarray(y, float), np.asarray(b_hat, float)
    if (h.ndim != 2 or not h.shape[1] or mh.shape != h.shape
            or y.shape != (len(h),) or b_hat.shape != (len(h),)
            or len(h) <= h.shape[1]):
        raise ValueError('Dimensões incompatíveis: H e m_H devem ser matrizes alinhadas.')
    if not all(np.isfinite(z).all() for z in (h, mh, y, b_hat)):
        raise ValueError('Valores ausentes ou não finitos.')
    n, p = h.shape
    initial = np.zeros(p) if start is None else np.asarray(start, float)
    if initial.shape != (p,) or not np.isfinite(initial).all():
        raise ValueError('start deve ser um vetor finito com um valor por coeficiente.')
    if not np.isin(y, [0, 1]).all() or not np.any(y == 1):
        raise ValueError('y deve ser binário e conter conversores.')
    if np.any((b_hat < 0) | (b_hat > 1)):
        raise ValueError('b_hat deve estar entre 0 e 1.')
    r = h-mh
    if np.linalg.matrix_rank(h) < p or np.linalg.matrix_rank(r) < p:
        raise RuntimeError('Jacobiano mal condicionado: base sem identificação (posto deficiente).')

    def transformed_outcome(theta):
        with np.errstate(over='raise', invalid='raise'):
            eta = -h@theta
            if not np.isfinite(eta).all() or np.max(np.abs(eta)) > 100:
                raise FloatingPointError('Extrapolação numérica na busca da raiz.')
            return y*np.exp(eta)

    def moment(theta):
        res = transformed_outcome(theta)-b_hat
        return (r*res[:, None]).mean(axis=0)

    def jac(theta):
        wy = transformed_outcome(theta)
        return -(r.T@(h*wy[:, None]))/n

    solution = root(moment, initial,
                    jac=jac, method='hybr', options={'xtol': 1e-10})
    norm = float(np.linalg.norm(moment(solution.x)))
    if not np.isfinite(norm) or (not solution.success and norm > 1e-7) or norm > 1e-6:
        raise RuntimeError(f'Raiz não convergiu: {solution.message}; norma={norm}')
    j = jac(solution.x)
    condition = float(np.linalg.cond(j))
    if not np.isfinite(condition) or condition > 1e10:
        raise RuntimeError('Jacobiano mal condicionado: revisar identificação/suporte/modelo.')
    psi = r*(transformed_outcome(solution.x)-b_hat)[:, None]
    omega = (psi.T@psi)/n
    invj = np.linalg.inv(j)
    cov = invj@omega@invj.T/n
    cov = (cov + cov.T)/2
    if not np.isfinite(cov).all():
        raise FloatingPointError('Covariância não finita: revisar identificação/escala.')
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return Estimate(solution.x.tolist(), se.tolist(), norm, condition,
                    cov.tolist(), np.linalg.svd(j, compute_uv=False).tolist())
