"""Experimentos reproduzíveis do manuscrito CQ-DML.

Dados integralmente sintéticos. A comparação Monte Carlo com nuisances fixas
é um teste de identidade/robustez, não um benchmark de ML em dados reais.

Uso: python experiments.py --reps 100 --n 30000 --demo-n 100000
Dependências: numpy, scipy, pandas, scikit-learn; matplotlib apenas para figuras.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from relative_dml import Estimate, fit_multiplicative_dml
from numpy.polynomial.legendre import leggauss

Array = NDArray[np.float64]
TRUE_THETA = np.array([-0.35, -0.20])
SEED = 20260905


def log_mgf(z: Array) -> Array:
    """log E[exp(z U)], U uniforme [-1,1], estável perto de zero."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    t = z[small]
    out[small] = t*t/6 - t**4/180 + t**6/2835
    t = np.abs(z[~small])
    out[~small] = t + np.log1p(-np.exp(-2*t)) - np.log(2*t)
    return out


def tilt_mean(z: Array) -> Array:
    """Média da uniforme exponencialmente inclinada."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    t = z[small]
    out[small] = t/3 - t**3/45 + 2*t**5/945
    t = z[~small]
    out[~small] = 1/np.tanh(t) - 1/t
    return out


def tilt_variance(z: Array) -> Array:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    t = z[small]
    out[small] = 1/3 - t*t/15 + 2*t**4/189
    t = z[~small]
    out[~small] = 1/(t*t) - 1/np.sinh(t)**2
    return out


def calibration_constant(theta: Array = TRUE_THETA) -> float:
    nodes, weights = leggauss(160)
    mean_unscaled = 0.0
    for x1 in (-1.0, 1.0):
        alpha = 0.9*x1 + 0.6*nodes
        beta = theta[0] + theta[1]*x1
        mean_unscaled += 0.5*np.sum(weights/2 * np.exp(
            1.1*x1 + 0.7*nodes + log_mgf(alpha+beta)-log_mgf(alpha)))
    return 0.06/mean_unscaled


SCALE = calibration_constant()


def generate(n: int, rng: np.random.Generator, theta: Array = TRUE_THETA):
    x = np.column_stack([rng.choice([-1.0, 1.0], n), rng.uniform(-1, 1, n)])
    alpha = 0.9*x[:, 0] + 0.6*x[:, 1]
    u = rng.uniform(0, 1, n)
    # |alpha| >= 0.3 neste desenho. Fórmula inversa da CDF.
    t = -1 + np.log1p(u*np.expm1(2*alpha))/alpha
    b = SCALE*np.exp(1.1*x[:, 0] + 0.7*x[:, 1])
    v = np.column_stack([np.ones(n), x[:, 0]])
    mu = b*np.exp(t*(v@theta))
    if not np.all((mu > 0) & (mu < 1)):
        raise ValueError('DGP produziu probabilidades inválidas.')
    y = rng.binomial(1, mu).astype(float)
    return x, t, y, b, alpha, mu


def fit_q(t: Array, y: Array, v: Array, alpha_hat: Array) -> Estimate:
    """Q contínuo no submodelo de inclinação exponencial, sem augmentation."""
    keep = y == 1
    tc, vc, z = t[keep], v[keep], alpha_hat[keep]
    nc = len(tc)
    if nc < 20:
        raise ValueError('Número insuficiente de conversores.')

    def obj(theta):
        eta = vc@theta
        return float(np.mean(log_mgf(z+eta)-tc*eta))

    def grad(theta):
        return np.mean(vc*(tilt_mean(z+vc@theta)-tc)[:, None], axis=0)

    sol = minimize(obj, np.zeros(v.shape[1]), jac=grad, method='BFGS',
                   options={'gtol': 1e-9, 'maxiter': 200})
    norm = float(np.linalg.norm(grad(sol.x)))
    if norm > 1e-6:
        raise RuntimeError(f'Q não convergiu: {sol.message}')
    r = vc*(tc-tilt_mean(z+vc@sol.x))[:, None]
    hess = vc.T@(vc*tilt_variance(z+vc@sol.x)[:, None])/nc
    inv = np.linalg.inv(hess)
    cov = inv@(r.T@r/nc)@inv/nc
    return Estimate(sol.x.tolist(), np.sqrt(np.diag(cov)).tolist(), norm,
                    float(np.linalg.cond(hess)))


def run_mc(reps: int, n: int, out: Path):
    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(reps):
        x, t, y, b, alpha, _ = generate(n, rng)
        v = np.column_stack([np.ones(n), x[:, 0]])
        mt = tilt_mean(alpha)
        modes = [('CQ-DML: b e m corretos', b, mt),
                 ('CQ-DML: apenas m correto', np.full(n, 0.06), mt),
                 ('CQ-DML: apenas b correto', b, np.zeros(n)),
                 ('CQ-DML: ambos incorretos', np.full(n, 0.06), np.zeros(n))]
        estimates = [(name, fit_multiplicative_dml(t,y,v,bh,mh)) for name,bh,mh in modes]
        estimates += [('Q: densidade correta', fit_q(t,y,v,alpha)),
                      ('Q: densidade incorreta', fit_q(t,y,v,np.zeros(n)))]
        for name, est in estimates:
            rows.append({'rep':rep, 'method':name, 'n':n, 'conversion':y.mean(),
                         **{f'theta{j}':est.theta[j] for j in range(2)},
                         **{f'se{j}':est.se[j] for j in range(2)},
                         'moment_norm':est.moment_norm})
    df = pd.DataFrame(rows)
    df.to_csv(out/'mc_raw.csv', index=False)
    summaries = []
    for name, sub in df.groupby('method', sort=False):
        row = {'method':name,'reps':reps, 'n':n, 'mean_conversion':sub.conversion.mean()}
        for j in range(2):
            delta=sub[f'theta{j}']-TRUE_THETA[j]
            row.update({f'mean{j}':sub[f'theta{j}'].mean(), f'bias{j}':delta.mean(),
                        f'mcse_bias{j}':sub[f'theta{j}'].std(ddof=1)/np.sqrt(reps),
                        f'rmse{j}':np.sqrt(np.mean(delta**2)),
                        f'coverage{j}':np.mean(np.abs(delta)<=1.96*sub[f'se{j}'])})
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'mc_summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    return summary


def run_ml_demo(n: int, out: Path):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    rng=np.random.default_rng(SEED+1)
    x,t,y,b,alpha,mu=generate(n,rng)
    nfolds=3
    bhat=np.zeros(n); mhat=np.zeros(n)
    for tr,te in KFold(nfolds,shuffle=True,random_state=SEED).split(x):
        reg=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=12,
                                        min_samples_leaf=150,l2_regularization=2,
                                        random_state=SEED)
        reg.fit(x[tr],t[tr]); mhat[te]=reg.predict(x[te])
        clf=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=12,
                                         min_samples_leaf=150,l2_regularization=2,
                                         random_state=SEED)
        clf.fit(np.column_stack([x[tr],t[tr]]),y[tr])
        bhat[te]=clf.predict_proba(np.column_stack([x[te],np.zeros(len(te))]))[:,1]
    v=np.column_stack([np.ones(n),x[:,0]])
    est=fit_multiplicative_dml(t,y,v,bhat,mhat)
    result={'n':n,'folds':nfolds,'seed':SEED+1,'conversion':float(y.mean()),
            'true_theta':TRUE_THETA.tolist(),'estimate':asdict(est),
            'b_rmse':float(np.sqrt(np.mean((bhat-b)**2))),
            'm_rmse':float(np.sqrt(np.mean((mhat-tilt_mean(alpha))**2))),
            'note':'Uma realização ilustrativa com nuisances estimadas por boosting; não é benchmark.'}
    (out/'ml_demo.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    return result


def analytic_checks(out: Path):
    from scipy.integrate import quad
    # DR de intervenções fixas: duas uniformes locais em [-1,1].
    def f(t): return float(np.exp(0.8*t-log_mgf(np.array(0.8)))/2)
    def fbad(t): return 0.5
    def mu(t): return 0.06*np.exp(-0.4*t)
    def mubad(t): return 0.09+0*t
    kernels=[(-0.6,-0.2),(0.2,0.6)]
    checks=[]
    for j,(lo,hi) in enumerate(kernels):
        true=quad(lambda t:mu(t)/(hi-lo),lo,hi,epsabs=1e-12)[0]
        for label,fh,mh in [('ambos corretos',f,mu),('apenas f correto',f,mubad),
                            ('apenas mu correto',fbad,mu),('ambos incorretos',fbad,mubad)]:
            expected=quad(lambda t:(mh(t)+f(t)/fh(t)*(mu(t)-mh(t)))/(hi-lo),lo,hi,epsabs=1e-12)[0]
            rem=quad(lambda t:(f(t)/fh(t)-1)*(mu(t)-mh(t))/(hi-lo),lo,hi,epsabs=1e-12)[0]
            checks.append({'kernel':j,'case':label,'truth':true,'expected_signal':expected,
                           'bias':expected-true,'product_remainder':rem})
            assert abs(expected-true-rem)<1e-10
            if label!='ambos incorretos': assert abs(expected-true)<1e-10
    pd.DataFrame(checks).to_csv(out/'stochastic_dr_checks.csv',index=False)
    # Q por densidade: identidade numérica, mesmo com baseline alto.
    for b in [0.02,0.06,0.25]:
        alpha=0.8; theta=-0.4; t=0.6; t0=-0.2
        log_r_q=(alpha+theta)*(t-t0)-alpha*(t-t0)
        assert abs(np.exp(log_r_q)-np.exp(theta*(t-t0)))<1e-12
    (out/'metadata.json').write_text(json.dumps({'seed':SEED,'scale':SCALE,
        'true_theta':TRUE_THETA.tolist(),'target_observed_conversion':0.06,
        'design':'X1 uniforme em {-1,+1}; X2 uniforme [-1,1]; alpha=.9X1+.6X2; T em [-1,1]',
        'baseline':'b=c exp(1.1 X1+0.7 X2); mu=b exp(T(theta0+theta1 X1))'},indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=100)
    parser.add_argument('--n',type=int,default=30000)
    parser.add_argument('--demo-n',type=int,default=100000)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    if args.reps<2 or args.n<100 or args.demo_n<100:
        parser.error('reps>=2 e tamanhos amostrais >=100 são necessários.')
    args.out.mkdir(parents=True,exist_ok=True)
    analytic_checks(args.out)
    run_mc(args.reps,args.n,args.out)
    run_ml_demo(args.demo_n,args.out)

if __name__=='__main__':
    main()
