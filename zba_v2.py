# -*- coding: utf-8 -*-
"""
ZAB neutra ZnCl2-NH4Cl — núcleo v2 (numpy puro).
Base: Clark, Latz e Horstmann (2017) + SI. Aproximações explícitas no fim do arquivo (PREMISSAS).

Estado por volume (mol/m3 de eletrólito): quasipartículas Q = [Zn~, NH3~, NH4~]  (Eqs. 21-23 do artigo)
    N = NH3~ + NH4~ ;  Cl = 2 Zn~ + NH4~   (eletroneutralidade, Eqs. 24-26)
Fontes (Tabela 5): Zn -> Zn~ + 2e- (eletrodo de Zn) ; 2 NH4~ + 2e- -> 2 NH3~ (GDE, ORR; inverso na OER).
Transporte: Nernst-Planck por espécie, potencial resolvido por corrente imposta, difusão implícita (bloco 3x3).
Precipitação: equilíbrio local (SI=0) com 4 sólidos; dissolve na carga.
"""
import math, time
import numpy as np

F = 96485.33212
R = 8.314462618
T = 298.15
RTF = R*T/F
LN10 = math.log(10.0)

# ---------------- espécies (mesma tabela/ordem do final5, 16 aquosas) ----------------
# (nome, M g/mol, z, D m2/s, lambda S m2/equiv, Vbar m3/mol)
SPECIES = [
    ("ZnCl+", 100.8,  1, 1.10e-9, 35.0e-4, -1.28e-6),
    ("ZnCl2",  136.3, 0, 1.10e-9,  0.0,    24.82e-6),
    ("ZnCl3-",171.7, -1, 1.10e-9, 35.0e-4,  53.9e-6),
    ("ZnCl4--",207.2,-2, 1.10e-9, 55.0e-4,  81.0e-6),
    ("Zn(NH3)2+",82.4, 2, 1.12e-9, 84.2e-4,  7.5e-6),
    ("Zn(NH3)2^2+",99.4,2,1.14e-9, 85.7e-4, 39.2e-6),
    ("Zn(NH3)3^2+",116.5,2,1.09e-9,81.5e-4, 70.9e-6),
    ("Zn(NH3)4^2+",133.5,2,0.83e-9,62.7e-4,102.6e-6),
    ("ZnCl3(NH3)-",188.8,-1,1.10e-9,41.3e-4, 86.4e-6),
    ("ZnCl(NH3)3+",151.9,1,1.10e-9,41.3e-4,  97.2e-6),
    ("H+",1.0,1,9.31e-9,349.8e-4,0.0),
    ("OH-",17.0,-1,5.26e-9,197.6e-4,-4.18e-6),
    ("NH4+",18.0,1,1.95e-9,73.4e-4,18.13e-6),
    ("NH3",17.0,0,1.94e-9,0.0,24.43e-6),
    ("Cl-",35.5,-1,2.03e-9,76.34e-4,17.79e-6),
    ("Zn2+",65.4,2,0.71e-9,53.0e-4,-24.3e-6),
]
NS = len(SPECIES)
IDX = {s[0]: i for i, s in enumerate(SPECIES)}
IZN, ICL, INH3, INH4, IH, IOH = IDX["Zn2+"], IDX["Cl-"], IDX["NH3"], IDX["NH4+"], IDX["H+"], IDX["OH-"]
Z_CH = np.array([s[2] for s in SPECIES], float)
D_SP = np.array([s[3] for s in SPECIES], float)
ZD = Z_CH*D_SP
Z2D = Z_CH**2*D_SP

# 12 log10(beta) da Tabela 1
LOG_BETA = [0.10, 0.06, 0.10, 0.30, 2.38, 4.88, 7.43, 9.65, 3.15, 7.90, 9.80, 13.96]
# expoentes de [Zn2+], [Cl-], [NH3], [H+] e log10(beta) por espécie
_EXP = {
    "ZnCl+": ((1,1,0,0), LOG_BETA[0]), "ZnCl2": ((1,2,0,0), LOG_BETA[1]),
    "ZnCl3-": ((1,3,0,0), LOG_BETA[2]), "ZnCl4--": ((1,4,0,0), LOG_BETA[3]),
    "Zn(NH3)2+": ((1,0,1,0), LOG_BETA[4]), "Zn(NH3)2^2+": ((1,0,2,0), LOG_BETA[5]),
    "Zn(NH3)3^2+": ((1,0,3,0), LOG_BETA[6]), "Zn(NH3)4^2+": ((1,0,4,0), LOG_BETA[7]),
    "ZnCl3(NH3)-": ((1,3,1,0), LOG_BETA[8]), "ZnCl(NH3)3+": ((1,1,3,0), LOG_BETA[9]),
    "H+": ((0,0,0,1), 0.0), "OH-": ((0,0,0,-1), -LOG_BETA[11]),
    "NH4+": ((0,0,1,1), LOG_BETA[10]), "NH3": ((0,0,1,0), 0.0),
    "Cl-": ((0,1,0,0), 0.0), "Zn2+": ((1,0,0,0), 0.0),
}
E = np.array([_EXP[s[0]][0] for s in SPECIES], float)            # (16,4)
LNB = np.array([_EXP[s[0]][1] for s in SPECIES], float)*LN10      # (16,)

# pesos das quantidades conservadas: Zn, Cl, NH3~, NH4~
W = np.zeros((4, NS))
for nm in ["ZnCl+","ZnCl2","ZnCl3-","ZnCl4--","Zn(NH3)2+","Zn(NH3)2^2+","Zn(NH3)3^2+",
           "Zn(NH3)4^2+","ZnCl3(NH3)-","ZnCl(NH3)3+","Zn2+"]:
    W[0, IDX[nm]] = 1.0
for nm, v in [("ZnCl+",1),("ZnCl2",2),("ZnCl3-",3),("ZnCl4--",4),("ZnCl3(NH3)-",3),("ZnCl(NH3)3+",1),("Cl-",1)]:
    W[1, IDX[nm]] = v
for nm, v in [("Zn(NH3)2+",1),("Zn(NH3)2^2+",2),("Zn(NH3)3^2+",3),("Zn(NH3)4^2+",4),
              ("ZnCl3(NH3)-",1),("ZnCl(NH3)3+",3),("NH3",1),("H+",-1),("OH-",1)]:
    W[2, IDX[nm]] = v
for nm, v in [("NH4+",1),("H+",1),("OH-",-1)]:
    W[3, IDX[nm]] = v
WABS = np.abs(W)
TAU = W[[0, 2, 3], :].T.copy()            # (16,3): contribuição de cada espécie às quasipartículas
DT_RHS = np.array([[1,0,0],[2,0,1],[0,1,0],[0,0,1]], float)   # dT/dQ

# ---------------- sólidos ----------------
SOLIDS = ["ZnO", "ZnCl2_4ZnOH2_H2O", "ZnOH2", "ZnCl2_2NH3"]    # mesma ordem do final5
LOG_KSP = {"ZnO": -16.7, "ZnCl2_4ZnOH2_H2O": -14.2, "ZnOH2": -17.0, "ZnCl2_2NH3": -6.42}
# CALIBRAÇÃO: com a Tabela 2 literal, o eletrólito C (0.5 M Zn, pH 8) fica ~10x supersaturado em Zn(OH)2, mas o artigo
# afirma que C está no limite de solubilidade. Deslocar log10 Ksp dos hidróxidos em +1.0 reproduz essa afirmação e as
# demais (1o sólido em A = simonkolleite; em B = ZnCl2.2NH3). Use 0.0 para a Tabela 2 literal.
KSP_SHIFT_OH = 1.0
_SHIFT = {"ZnO": KSP_SHIFT_OH, "ZnOH2": KSP_SHIFT_OH, "ZnCl2_4ZnOH2_H2O": KSP_SHIFT_OH, "ZnCl2_2NH3": 0.0}
# SI = log10 Q - log10 Ksp, com Q por Zn: Zn*OH^m*Cl^c (simonkolleite: m=1.6, c=0.4) — ver PREMISSAS
def _si_terms():
    a, k = {}, {}
    # OH = 1/(beta_w h): ln OH = -ln beta_w - ln h
    for nm, (ez, ecl, enh, eoh) in {"ZnO": (1,0,0,2), "ZnOH2": (1,0,0,2),
                                    "ZnCl2_4ZnOH2_H2O": (1,0.4,0,1.6), "ZnCl2_2NH3": (1,2,2,0)}.items():
        a[nm] = np.array([ez, ecl, enh, -eoh], float)
        k[nm] = -eoh*LOG_BETA[11] - (LOG_KSP[nm] + _SHIFT[nm])   # constante em log10
    return a, k
SI_A, SI_K = _si_terms()
# efeito de cada sólido (por mol de sólido) em Q=[Zn~,NH3~,NH4~] — Tabela 5
NU = {"ZnO": np.array([-1,-2,2.]), "ZnOH2": np.array([-1,-2,2.]),
      "ZnCl2_4ZnOH2_H2O": np.array([-5,-8,8.]), "ZnCl2_2NH3": np.array([-1,-2,0.])}
# volumes molares (m3/mol): ZnO 14.5, Zn(OH)2 32.6, simonkolleite ~172, ZnCl2.2NH3 ~85 (estimado)
VM = {"ZnO": 14.5e-6, "ZnOH2": 32.6e-6, "ZnCl2_4ZnOH2_H2O": 172.0e-6, "ZnCl2_2NH3": 85.0e-6}
V_ZN = 9.16e-6
ZT_FLOOR = 1e-3          # mol/m3 (1 uM): piso numérico do Zn~; desvio de massa de Zn <= 1e-5 relativo

# ---------------- especiação vetorizada ----------------
def speciate(Q, y0=None, tol=1e-11, maxit=80, want_dy=False):
    """Q (n,3) mol/m3 -> y=ln([Zn2+],[Cl-],[NH3],[H+]) em mol/L, c (n,16) mol/m3, S (n,16,3) dc/dQ, ok (n,)."""
    Q = np.atleast_2d(np.asarray(Q, float))
    n = len(Q)
    q = Q/1000.0
    Zt, A3, A4 = q[:,0], q[:,1], q[:,2]
    Clt = 2*Zt + A4
    tgt = np.stack([Zt, Clt, A3, A4], 1)
    if y0 is None:
        y = _cold_start(Zt, Clt, A3, A4)
    else:
        y = np.array(y0, float).reshape(n,4).copy()

    def resid(yy):
        c = np.exp(LNB + yy@E.T)
        Fv = c@W.T - tgt
        mag = c@WABS.T + np.abs(tgt) + 1e-300
        return c, Fv, mag, np.max(np.abs(Fv)/mag, axis=1)

    c, Fv, mag, err = resid(y)
    for it in range(maxit):
        act = err > tol
        if not act.any():
            break
        J = (c[act][:,None,:]*W[None,:,:]) @ E            # (m,4,4)
        sc = mag[act]
        Jn = J/sc[:,:,None]
        rhs = -Fv[act]/sc
        try:
            dy = np.linalg.solve(Jn, rhs[:,:,None])[:,:,0]
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(Jn[0], rhs[0], rcond=None)[0][None,:].repeat(act.sum(),0)
        dy = np.clip(dy, -3.0, 3.0)
        ia = np.where(act)[0]
        best_err = err[ia].copy(); best_y = y[ia].copy()
        done = np.zeros(len(ia), bool)
        for damp in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03):
            yt = y[ia] + damp*dy
            ct, Ft, mt, et = resid_sub(yt, tgt[ia])
            better = (et < best_err) & ~done
            best_err[better] = et[better]; best_y[better] = yt[better]; done |= better
            if done.all():
                break
        y[ia] = best_y
        c, Fv, mag, err = resid(y)
    ok = err < 1e-8
    S = None
    dydq = None
    if want_dy or True:
        J = (c[:,None,:]*W[None,:,:]) @ E
        try:
            dydq = np.linalg.solve(J, np.broadcast_to(DT_RHS, (n,4,3)).copy())
        except np.linalg.LinAlgError:
            dydq = np.zeros((n,4,3)); ok &= False
        S = c[:,:,None]*np.einsum('sk,nkp->nsp', E, dydq)
    return y, c*1000.0, S, ok, dydq

def _cold_start(Zt, Clt, A3, A4):
    n = len(Zt)
    nh3 = np.maximum(A3 - 3.5*Zt, 0.02*np.abs(A3) + 1e-7)
    h = np.clip(np.maximum(A4, 1e-9)/(10**LOG_BETA[10]*nh3), 1e-14, 1.0)
    y = np.empty((n,4))
    y[:,0] = np.log(np.maximum(Zt*0.05, 1e-14))
    y[:,1] = np.log(np.maximum(Clt*0.6, 1e-14))
    y[:,2] = np.log(nh3)
    y[:,3] = np.log(h)
    return y

def resid_sub(yy, tgt):
    c = np.exp(LNB + yy@E.T)
    Fv = c@W.T - tgt
    mag = c@WABS.T + np.abs(tgt) + 1e-300
    return c, Fv, mag, np.max(np.abs(Fv)/mag, axis=1)

def si_all(y):
    """SI (n,4) na ordem SOLIDS, a partir de y."""
    out = np.empty((len(y), 4))
    for k, nm in enumerate(SOLIDS):
        out[:,k] = (y@SI_A[nm])/LN10 + SI_K[nm]
    return out

def init_composition(zncl2_M, nh4cl_M, pH):
    """Q0 (mol/m3) com NH4OH ajustado ao pH: Zn~, NH4~ fixos; NH3~ resulta do pH."""
    Zt = zncl2_M; A4 = nh4cl_M; Clt = 2*Zt + A4; h = 10.0**(-pH)
    y = np.array([math.log(Zt*0.05), math.log(Clt*0.6), math.log(max(A4,1e-6)/(10**LOG_BETA[10]*h))])
    rows = [0, 1, 3]
    for it in range(200):
        yy = np.array([y[0], y[1], y[2], math.log(h)])
        c = np.exp(LNB + E@yy)
        Fv = np.array([W[0]@c - Zt, W[1]@c - Clt, W[3]@c - A4])
        mag = np.array([WABS[0]@c + Zt, WABS[1]@c + Clt, WABS[3]@c + A4])
        if np.max(np.abs(Fv)/mag) < 1e-12:
            break
        J = (c[None,:]*W[rows])@E[:, :3]
        dy = np.linalg.solve(J/mag[:,None], -Fv/mag)
        y = y + np.clip(dy, -2, 2)
    else:
        raise ArithmeticError("init_composition não convergiu")
    A3 = float(W[2]@c)
    return np.array([Zt, A3, A4])*1000.0

# ---------------- modelo 1D ----------------
class Model:
    def __init__(self, setup=None):
        s = dict(setup or {})
        self.setup = s
        mm = 1e-3
        self.zn_model = s.get("zn_model", "paste")           # "paste" | "foil"
        paste = self.zn_model == "paste"
        self.L_ZN = float(s.get("zn_mm", 5.0))*mm if paste else 0.0
        self.L_SEP = float(s.get("separator_mm", 0.5))*mm
        self.L_GDE = float(s.get("gde_mm", 0.5))*mm
        n_zn = int(s.get("n_zn", 16)) if paste else 0
        n_sep = int(s.get("n_sep", 8 if self.L_SEP < 2*mm else 24))
        n_gde = int(s.get("n_gde", 4))
        # malha: Zn uniforme, separador com refino nas duas pontas, GDE uniforme
        ss = np.linspace(0, 1, n_sep+1)
        g = 1.6
        e_sep = self.L_SEP*(1 + np.tanh(g*(2*ss-1))/math.tanh(g))/2
        dx = []
        if n_zn: dx += [self.L_ZN/n_zn]*n_zn
        dx += list(np.diff(e_sep))
        dx += [self.L_GDE/n_gde]*n_gde
        self.dx = np.array(dx)
        self.region = ["ZN"]*n_zn + ["SEP"]*n_sep + ["GDE"]*n_gde
        self.N = len(self.dx)
        self.n_zn, self.n_sep, self.n_gde = n_zn, n_sep, n_gde
        self.iz = np.arange(0, n_zn)
        self.ig = np.arange(n_zn+n_sep, self.N)
        self.dxf = 0.5*(self.dx[:-1] + self.dx[1:])
        self.x = np.cumsum(self.dx) - 0.5*self.dx
        # porosidade do eletrólito (estática) e Bruggeman
        self.beta = float(s.get("bruggeman", 1.5))
        eps = np.empty(self.N)
        for i, r in enumerate(self.region):
            eps[i] = {"ZN": float(s.get("eps_e_zn", 0.30)), "SEP": float(s.get("eps_e_sep", 0.90)),
                      "GDE": float(s.get("eps_e_gde", 0.25))}[r]
        self.eps_e = eps
        self.epsb_f = 0.5*(eps[:-1]**self.beta + eps[1:]**self.beta)
        # zinco
        self.eps_zn0 = float(s.get("eps_zn0", 0.40))
        self.d_zn = float(s.get("d_zn_um", 100.0))*1e-6
        self.delta_c = float(s.get("delta_pass_um", 5.0))*1e-6
        self.eps_zn = np.zeros(self.N)
        if paste:
            self.eps_zn[self.iz] = self.eps_zn0
        self.foil_rf = float(s.get("foil_rf", 1.0))
        # cinética (calibrável) — ver PREMISSAS
        self.i0_zn = float(s.get("i0_zn", 2*F*8.0e-6))        # A/m2 (k0 do SI x 2F)
        self.i0_orr = float(s.get("i0_orr", 1.86e-6))         # A/m2 de área ativa (calibrado: V=0.90 V @1 mA/cm2, eletrólito C)
        self.i0_oer = float(s.get("i0_oer", 4.49e-3))         # A/m2 de área ativa (calibrado: V=2.25 V @2.5 mA/cm2, célula A)
        self.alpha_zn = 0.5
        self.alpha_orr = float(s.get("alpha_orr", 0.5))
        self.alpha_oer = float(s.get("alpha_oer", 0.5))
        self.A_gde = float(s.get("A_gde_sp", 4.5e3))
        self.pO2 = float(s.get("pO2_atm", 0.21))
        self.solids_on = [k for k in SOLIDS if k in s.get("solids", SOLIDS)]
        # estado
        zt = float(s.get("zncl2_M", 0.5)); nh = float(s.get("nh4cl_M", 1.6))
        self.initial_pH = float(s.get("initial_pH", 8.0))
        Q0 = init_composition(zt, nh, self.initial_pH)
        self.Q = np.tile(Q0, (self.N, 1))
        self.s = np.zeros((4, self.N))                    # sólidos, mol/m3 de volume total
        self.y = None
        self._respec()
        self.t = 0.0; self.cap = 0.0; self.energy = 0.0   # Ah/m2, Wh/m2
        self.V = float('nan')
        self.dt = float(s.get("dt0", 1.0)); self.dt_max = float(s.get("dt_max", 120.0))
        self.dpH_max = float(s.get("dpH_max", 1.0))
        self.history = []
        self.n_steps = 0; self.n_rej = 0
        self.stop_flag = False
        _g = float(self.pH()[self.ig].mean())
        self.pH_min_run = _g; self.pH_max_run = _g
        self._equilibrate_initial(zt, nh)
        self.inv0 = self.invariants()

    # ----- utilitários -----
    def _respec(self):
        y, c, S, ok, dydq = speciate(self.Q, self.y)
        if not ok.all():
            y2, c2, S2, ok2, dydq2 = speciate(self.Q[~ok], None)
            y[~ok], c[~ok], S[~ok], ok[~ok] = y2, c2, S2, ok2
            if not ok.all():
                raise ArithmeticError("especiação não convergiu")
        self.y, self.c, self.S = y, c, S

    def pH(self):
        return -np.log10(np.maximum(self.c[:, IH]/1000.0, 1e-30))

    def solids_vol(self):
        return sum(VM[k]*self.s[i] for i, k in enumerate(SOLIDS))

    def invariants(self):
        dx = self.dx; ee = self.eps_e
        zn = np.dot(dx, ee*self.Q[:,0]) + np.dot(dx, self.eps_zn)/V_ZN
        cl = np.dot(dx, ee*(2*self.Q[:,0] + self.Q[:,2]))
        nn = np.dot(dx, ee*(self.Q[:,1] + self.Q[:,2]))
        s = {k: np.dot(dx, self.s[i]) for i, k in enumerate(SOLIDS)}
        zn += 5*s["ZnCl2_4ZnOH2_H2O"] + s["ZnO"] + s["ZnOH2"] + s["ZnCl2_2NH3"]
        cl += 2*s["ZnCl2_4ZnOH2_H2O"] + 2*s["ZnCl2_2NH3"]
        nn += 2*s["ZnCl2_2NH3"]
        return np.array([zn, cl, nn])

    def _equilibrate_initial(self, zt, nh):
        """Eletrólito preparado: precipita o excedente e reajusta o pH com NH4OH (só NH3~ muda)."""
        for it in range(12):
            self._flash(self.Q, self.s); self._respec()
            q0 = self.Q[0]/1000.0
            A3 = init_composition(q0[0], q0[2], self.initial_pH)[1]
            dA3 = A3 - self.Q[0,1]
            self.Q[:,1] += dA3
            self._respec()
            if abs(dA3) < 1e-6*max(1.0, abs(A3)) and np.max(np.abs(self.pH()-self.initial_pH)) < 1e-6:
                break
        self._flash(self.Q, self.s); self._respec()

    # ----- precipitação / dissolução em equilíbrio -----
    def _flash(self, Q, s, tol_si=2e-6):
        ee = self.eps_e
        y = self.y if self.y is not None else speciate(Q)[0]
        SIc = si_all(y)
        for sweep in range(3):
            moved = False
            for k, nm in enumerate(SOLIDS):
                if nm not in self.solids_on:
                    continue
                SI = SIc[:, k]
                have = s[k] > 1e-9
                idx = np.where((SI > tol_si) | ((SI < -tol_si) & have))[0]
                if len(idx) == 0:
                    continue
                nu = NU[nm]
                Qi = Q[idx].copy(); yi = y[idx].copy()
                se = s[k][idx]/ee[idx]                           # sólido na base eletrólito
                hi = np.where(SI[idx] > 0, 0.99*Qi[:,0]/abs(nu[0]), 0.0)
                lo = np.where(SI[idx] < 0, -se, 0.0)
                xi = np.zeros(len(idx))
                a = SI_A[nm]
                for it in range(14):
                    Qt = Qi + xi[:,None]*nu
                    yt, ct, St, okt, dydq = speciate(Qt, yi)
                    f = (yt@a)/LN10 + SI_K[nm]
                    f = np.where(okt, f, -1.0)                  # falha -> trata como supersaturação negativa
                    lo = np.where(f > 0, xi, lo); hi = np.where(f <= 0, xi, hi)
                    d = np.einsum('k,nkp,p->n', a, dydq, nu)/LN10/1000.0
                    with np.errstate(all='ignore'):
                        new = xi - f/d
                    bad = ~np.isfinite(new) | (new <= lo) | (new >= hi)
                    new = np.where(bad, 0.5*(lo+hi), new)
                    conv = (np.abs(f) < 2e-7) | ((hi - lo) < 1e-12*np.maximum(1.0, np.abs(hi)+np.abs(lo)))
                    xi = np.where(conv, xi, new)
                    yi = np.where(okt[:,None], yt, yi)
                    if conv.all():
                        break
                Q[idx] = Qi + xi[:,None]*nu
                yi, _c, _S, _ok, _d = speciate(Q[idx], yi)
                s[k][idx] = np.maximum(s[k][idx] + ee[idx]*xi, 0.0)
                moved = True
                y[idx] = yi
                SIc[idx] = si_all(yi)
            if not moved:
                break
        self.y = y
        self._flash_moved = moved

    # ----- zinco: área ativa e passivação -----
    def _zn_area(self, I):
        n = self.N; az = np.zeros(n)
        if self.zn_model == "foil":
            az[self.n_zn] = self.foil_rf/self.dx[self.n_zn]
            return az, np.ones(n)
        iz = self.iz
        eps = np.maximum(self.eps_zn[iz], 1e-3)
        A = 6.0/self.d_zn*self.eps_zn0**(1/3)*eps**(2/3)
        npart = 6.0*self.eps_zn0/(math.pi*self.d_zn**3)
        Vs = self.solids_vol()[iz]
        r = 0.5*self.d_zn*(eps/self.eps_zn0)**(1/3)
        Rr = (r**3 + 3.0*Vs/(4*math.pi*npart))**(1/3)
        delta = Rr - r
        f = np.clip(1.0 - (delta/self.delta_c)**2, 0.0, 1.0) if I > 0 else np.ones(len(iz))
        az[iz] = A*f
        self._delta = delta
        self._fmean = float(np.mean(f))
        return az, f

    # ----- potencial do eletrólito (inclui potencial de difusão) -----
    def _phi_e(self, c, jf):
        cf = 0.5*(c[:-1] + c[1:])
        gr = (c[1:] - c[:-1])/self.dxf[:,None]
        G = cf@Z2D
        dif = gr@ZD
        dphi = -RTF*((jf[1:-1]/F)/self.epsb_f + dif)/np.maximum(G, 1e-12)*self.dxf
        return np.concatenate([[0.0], np.cumsum(dphi)])

    def _solve_phis(self, base, w, ai, ac, target):
        """Σ w (exp(ai(φ-b)/RTF) - exp(-ac(φ-b)/RTF)) = target ; Newton monótono."""
        m = w > 0
        if not m.any():
            return None
        b = base[m]; ww = w[m]
        phi = float(np.dot(ww, b)/ww.sum())
        for it in range(200):
            e1 = np.exp(np.clip(ai*(phi-b)/RTF, -80, 80)); e2 = np.exp(np.clip(-ac*(phi-b)/RTF, -80, 80))
            g = np.dot(ww, e1 - e2) - target
            dg = np.dot(ww, ai/RTF*e1 + ac/RTF*e2)
            if abs(g) < 1e-12*max(abs(target), 1e-12):
                break
            step = np.clip(-g/dg, -0.25, 0.25)
            phi += step
        return phi

    def kinetics(self, I):
        """I em A/m2 (>0 descarga). Devolve rz, rg (A/m3), V, diagnósticos."""
        c = self.c; cM = c/1000.0; dx = self.dx; n = self.N
        Uz = -0.762 + RTF/2*np.log(np.maximum(cM[:,IZN], 1e-30))
        Ug = 1.229 + RTF/4*math.log(self.pO2) + RTF*np.log(np.maximum(cM[:,IH], 1e-30))
        az, fp = self._zn_area(I)
        ag = np.zeros(n); ag[self.ig] = self.A_gde
        zmask = az > 0
        if I == 0.0:
            wz = az*dx; wg = ag*dx
            V = float(np.dot(wg, Ug)/wg.sum() - np.dot(wz, Uz)/wz.sum())
            return np.zeros(n), np.zeros(n), V, {"eta_z": 0.0, "eta_g": 0.0}
        if az.sum() <= 0:
            return np.zeros(n), np.zeros(n), -1e9, {"passivated": True}
        i0g = self.i0_orr if I > 0 else self.i0_oer
        wz0 = az*dx; wg0 = ag*dx
        rz = I*wz0/wz0.sum(); rg = -I*wg0/wg0.sum()
        for it in range(3):
            jf = np.concatenate([[0.0], np.cumsum((rz + rg)*dx)])
            phi = self._phi_e(c, jf)
            wz = self.i0_zn*az*dx; wg = i0g*ag*dx
            ag_ = self.alpha_orr if I > 0 else self.alpha_oer
            pz = self._solve_phis(phi + Uz, wz, self.alpha_zn, self.alpha_zn, I)
            pg = self._solve_phis(phi + Ug, wg, ag_, ag_, -I)
            if pz is None or pg is None:
                return np.zeros(n), np.zeros(n), -1e9, {"passivated": True}
            ez = np.clip((pz - phi - Uz)/RTF*self.alpha_zn, -80, 80)
            eg = np.clip((pg - phi - Ug)/RTF*ag_, -80, 80)
            rz = self.i0_zn*az*(np.exp(ez) - np.exp(-ez))
            rg = i0g*ag*(np.exp(eg) - np.exp(-eg))
        V = pg - pz
        return rz, rg, float(V), {"eta_z": float(np.dot(az*dx, pz-phi-Uz)/max((az*dx).sum(),1e-30)),
                                  "eta_g": float(np.dot(ag*dx, pg-phi-Ug)/(ag*dx).sum()),
                                  "ohm": float(phi[self.ig].mean() - phi[:max(self.n_zn,1)].mean()) if I != 0 else 0.0}

    # ----- um passo -----
    def _transport_matrix(self, jf):
        c, S = self.c, self.S
        n = self.N
        cf = 0.5*(c[:-1] + c[1:]); Sf = 0.5*(S[:-1] + S[1:])
        G = np.maximum(cf@Z2D, 1e-12)
        A = np.einsum('iq,i,fip->fqp', TAU, D_SP, Sf)
        Bv = np.einsum('i,fip->fp', ZD, Sf)
        M = np.einsum('iq,fi->fq', TAU*ZD[:,None], cf)/G[:,None]
        K = self.epsb_f[:,None,None]*(A - M[:,:,None]*Bv[:,None,:])/self.dxf[:,None,None]
        m = M*(jf[1:-1]/F)[:,None]
        return K, m

    def advance(self, I, dt, kin=None):
        n = self.N; dx = self.dx; ee = self.eps_e
        rz, rg, V, info = kin if kin is not None else self.kinetics(I)
        src = np.zeros((n, 3))
        src[:,0] = rz/(2*F)
        src[:,1] = -rg/F
        src[:,2] = rg/F
        jf = np.concatenate([[0.0], np.cumsum((rz + rg)*dx)])
        K, m = self._transport_matrix(jf)
        Am = np.zeros((3*n, 3*n)); rhs = np.zeros(3*n)
        Id = np.eye(3)
        for i in range(n):
            sl = slice(3*i, 3*i+3)
            Dg = ee[i]*dx[i]/dt*Id
            fl = np.zeros(3); fr = np.zeros(3)
            if i > 0:
                Dg = Dg + K[i-1]; Am[sl, 3*i-3:3*i] = -K[i-1]; fl = m[i-1]
            if i < n-1:
                Dg = Dg + K[i];   Am[sl, 3*i+3:3*i+6] = -K[i];   fr = m[i]
            Am[sl, sl] = Dg
            rhs[sl] = ee[i]*dx[i]/dt*self.Q[i] + (fl - fr) + dx[i]*src[i]
        Qn = np.linalg.solve(Am, rhs).reshape(n, 3)
        return Qn, V, info, rz

    def step(self, I):
        """Avança um passo adaptativo. Devolve (V, info)."""
        kin = self.kinetics(I)
        V = kin[2]
        if V < -1e8:
            return V, kin[3]
        while True:
            dt = self.dt
            snap = (self.Q.copy(), self.s.copy(), self.eps_zn.copy(), self.y.copy())
            ph0 = self.pH()
            try:
                Qn, V, info, rz = self.advance(I, dt, kin)
                if not np.all(np.isfinite(Qn)):
                    raise ArithmeticError("Q não finito")
                Qn[:,0] = np.maximum(Qn[:,0], ZT_FLOOR)       # piso numérico de Zn~ (volume esgotado)
                self.Q = Qn
                if self.zn_model == "paste":
                    self.eps_zn[self.iz] = np.maximum(self.eps_zn[self.iz] - V_ZN*rz[self.iz]/(2*F)*dt, 0.0)
                self.y = np.clip(self.y, -60, 20)
                y, c, S, ok, _ = speciate(self.Q, self.y)
                if not ok.all():
                    y2, c2, S2, ok2, _ = speciate(self.Q[~ok], None)
                    if not ok2.all():
                        raise ArithmeticError("especiação")
                    y[~ok] = y2
                self.y = y
                self._flash(self.Q, self.s)
                self._respec()
                dph = float(np.max(np.abs(self.pH() - ph0)))
                if dph > self.dpH_max and dt > 1e-3:
                    raise ArithmeticError("dpH")
            except (ArithmeticError, np.linalg.LinAlgError):
                self.Q, self.s, self.eps_zn, self.y = snap
                self._respec()
                self.dt = max(dt*0.5, 1e-4); self.n_rej += 1
                if dt <= 1e-4:
                    raise
                continue
            break
        self.t += dt; self.n_steps += 1
        self.cap += I*dt/3600.0
        self.energy += I*V*dt/3600.0
        self.V = V
        _g = float(self.pH()[self.ig].mean())
        self.pH_min_run = min(self.pH_min_run, _g); self.pH_max_run = max(self.pH_max_run, _g)
        self.dt = min(self.dt_max, dt*(1.4 if dph < 0.5*self.dpH_max else 1.0))
        return V, info

    # ----- protocolo -----
    def run_protocol(self, segments, rec_every_s=60.0, callback=None, max_steps=2_000_000, cb_wall_s=0.25):
        """segments: lista de dict(mode='discharge'|'charge'|'rest', I_mA_cm2, t_max_h, V_cut, q_max_mAh_cm2).
        callback(model, k, nseg, frac, V, cpu_s) é chamado a cada cb_wall_s s de relógio; stop_flag interrompe."""
        t0 = time.time(); next_rec = self.t; last_cb = 0.0; self.stop_flag = False
        out = []
        for seg in segments:
            mode = seg.get("mode", "discharge")
            I = seg.get("I_mA_cm2", 1.0)*10.0*(1 if mode == "discharge" else -1 if mode == "charge" else 0)
            t_end = self.t + seg.get("t_max_h", 1e9)*3600.0
            vcut = seg.get("V_cut", 0.6 if mode == "discharge" else 2.6 if mode == "charge" else None)
            q_end = self.cap + (seg.get("q_max_mAh_cm2", 1e9)*10.0)*(1 if I >= 0 else -1)   # mAh/cm2 -> Ah/m2: x10
            reason = "tempo"
            seg_t0 = self.t; seg_c0 = self.cap; seg_e0 = self.energy
            while True:
                if self.stop_flag: reason = "interrompido"; break
                if self.t >= t_end: reason = "tempo"; break
                if I > 0 and self.cap >= q_end: reason = "capacidade"; break
                if I < 0 and self.cap <= q_end: reason = "capacidade"; break
                dt_left = t_end - self.t
                self.dt = min(self.dt, dt_left)
                V, info = self.step(I)
                if V < -1e8:
                    reason = "passivação (Zn sem área ativa)"; break
                if vcut is not None and ((I > 0 and V < vcut) or (I < 0 and V > vcut)):
                    reason = f"V_cut {vcut} V"; break
                if self.t >= next_rec:
                    self.history.append(self.record(I, V, info)); next_rec = self.t + rec_every_s
                if callback and time.time() - last_cb >= cb_wall_s:
                    last_cb = time.time()
                    frac = min(1.0, (self.t - seg_t0)/max(t_end - seg_t0, 1e-9))
                    callback(self, len(out), len(segments), frac, V, time.time() - t0)
                if self.n_steps > max_steps: reason = "max_steps"; break
            if self.stop_flag: out.append({"mode": mode, "I_mA_cm2": I/10.0, "t_h": (self.t-seg_t0)/3600.0, "cap_mAh_cm2": (self.cap-seg_c0)/10.0, "E_Wh_m2": self.energy-seg_e0, "V_end": self.V, "reason": reason, "pH_gde": float(self.pH()[self.ig].mean()), "pH_zn": float(self.pH()[:self.n_zn+1].mean())}); break
            out.append({"mode": mode, "I_mA_cm2": I/10.0, "t_h": (self.t-seg_t0)/3600.0,
                        "cap_mAh_cm2": (self.cap-seg_c0)/10.0, "E_Wh_m2": self.energy-seg_e0,
                        "V_end": self.V, "reason": reason, "pH_gde": float(self.pH()[self.ig].mean()),
                        "pH_zn": float(self.pH()[:self.n_zn+1].mean())})
        self.cpu_s = time.time() - t0
        return out

    def record(self, I, V, info):
        ph = self.pH()
        return {"t_h": self.t/3600.0, "V": V, "I": I/10.0, "pH_gde": float(ph[self.ig].mean()),
                "pH_min": float(ph.min()), "pH_max": float(ph.max()),
                "pH_zn": float(ph[:max(self.n_zn,1)].mean()),
                "solids_mol_m2": [float(np.dot(self.dx, self.s[k])) for k in range(4)],
                "eps_zn_mean": float(self.eps_zn[self.iz].mean()) if self.n_zn else 1.0,
                "cap_mAh_cm2": self.cap/10.0,
                "delta_um": (float(np.max(self._delta))*1e6 if (self.n_zn and hasattr(self, "_delta")) else 0.0),
                "f_pass": float(getattr(self, "_fmean", 1.0)),
                "dt": self.dt, **{k: v for k, v in info.items() if isinstance(v, float)}}

    def snapshot(self):
        """Cópia leve do estado espacial para gráficos (chamar na mesma thread que avança o modelo)."""
        zf = lambda idx: {SPECIES[k][0]: float(v) for k, v in enumerate(
            (self.c[idx]*W[0]).mean(0)/max((self.c[idx]*W[0]).mean(0).sum(), 1e-30))}
        return {"x": list(self.x*1000.0), "edges": list(np.cumsum(self.dx)*1000.0),
                "n_zn": self.n_zn, "n_sep": self.n_sep, "pH": [float(v) for v in self.pH()],
                "Zt": [float(v) for v in self.Q[:,0]/1000.0], "A3": [float(v) for v in self.Q[:,1]/1000.0],
                "A4": [float(v) for v in self.Q[:,2]/1000.0], "z_gde": zf(self.ig),
                "z_zn": zf(np.arange(0, max(self.n_zn, 1)))}

    def energy_density_WhL(self):
        """Wh por litro de volume total da célula (Zn + separador + GDE)."""
        return self.energy/self.dx.sum()/1000.0

def _demo():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "C"
    E = {"A": (0.51, 2.34, 6.0, None), "B": (0.26, 5.0, 7.0, None), "C": (0.5, 1.6, 8.0, None), "D": (0.5, 2.0, 7.0, ["ZnO"])}
    if which == "cellA":      # Célula A do artigo: folha de Zn, separador 30 mm, 5 mA/cm2 x4 h / 2.5 mA/cm2 x8 h
        m = Model({"zncl2_M": 0.51, "nh4cl_M": 2.34, "initial_pH": 6.0, "zn_model": "foil", "separator_mm": 30.0})
        segs = []
        for _ in range(3):
            segs += [{"mode": "discharge", "I_mA_cm2": 5.0, "t_max_h": 4, "V_cut": 0.0},
                     {"mode": "charge", "I_mA_cm2": 2.5, "t_max_h": 8, "V_cut": 4.0}]
    else:                     # célula padrão de otimização: Zn 5 mm, separador 0.5 mm, GDE 0.5 mm, 1 mA/cm2 até 0.6 V
        z, nh, pH, sol = E[which]
        st = {"zncl2_M": z, "nh4cl_M": nh, "initial_pH": pH, "dt_max": 300.0}
        if sol: st["solids"] = sol
        m = Model(st)
        segs = [{"mode": "discharge", "I_mA_cm2": 1.0, "t_max_h": 2000, "V_cut": 0.6}]
    for r in m.run_protocol(segs, rec_every_s=3600.0):
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("CPU %.1fs | passos %d | rejeitados %d | E_dens %.1f Wh/L | deriva invariantes %s" %
          (m.cpu_s, m.n_steps, m.n_rej, m.energy_density_WhL(), ((m.invariants()-m.inv0)/m.inv0).round(10)))

if __name__ == "__main__":
    _demo()

# PREMISSAS (estáticas salvo indicação):
#  - Concentrações no lugar de atividades (I ~ 3-6 M): erro conhecido nas constantes e no Nernst.
#  - log10 Ksp dos hidróxidos deslocado em +1.0 (KSP_SHIFT_OH): calibração, não valor da Tabela 2.
#  - Ksp simonkolleite em base por-Zn: Zn*OH^1.6*Cl^0.4 (reproduz o cruzamento pH~7.5 a Cl~3 M do artigo).
#  - Volume molar ZnCl2.2NH3 ~85 cm3/mol (estimado); eps_e fixo por região; sem inundação do GDE e sem balanço de H2O.
#  - Passivação: casca de sólido sobre partícula esférica de Zn; ativa=1-(delta/5um)^2 (forma assumida).
#  - O2 no GDE: pO2 constante (dinâmico apenas se o usuário alterar pO2_atm).
#  - i0_orr / i0_oer: calibrados em curvas digitalizadas do artigo (Fig. 5/8), não medidos.
