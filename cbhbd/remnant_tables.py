"""
Tabulated single-star remnant outcomes from COMPAS.

This module lets cBHBd draw its first-generation (1G) black holes from tables of single-star outcomes computed with
COMPAS (``scripts/make_compas_remnant_tables.py``), so that cluster and isolated-binary populations can share the same
stellar evolution and supernova physics. Use it by passing the name of a table as the IFMR, e.g.

    CBHBD(M0=1e6, rh0=1, Z=0.001, ifmr="compas-maltsev-balanced")

Available tables are listed in ``available_tables()``. Each table stores, for a grid of metallicities and ZAMS masses,
several independent COMPAS realisations, so stochastic prescriptions (e.g. the partial-fallback branch of
MALTSEV2024) are sampled rather than averaged. Unlike the ssptools IFMRs, a star above the lowest black-hole-forming
mass can still form a neutron star, so the "islands of explodability" of compactness-based prescriptions are kept.

Options (set through ``CBHBD``):

- ``keep_envelope`` (default False): if True, BHs formed by core collapse also accrete the hydrogen envelope, scaled
  by the fallback fraction. COMPAS sets the BH mass to the He-core mass for MALTSEV2024, i.e. assumes that the
  envelope is lost.
- ``kick_model`` (default "auto"): "fallback" draws Maxwellian kicks with dispersion ``sigmans`` scaled by
  (1 - fallback fraction), as in the default cBHBd model. BHs from pulsational pair-instability supernovae receive no
  kick, as in COMPAS by default. "reduced" draws Maxwellian kicks with dispersion ``sigmans`` scaled by
  1.4 Msun / M_BH (momentum-conserving kicks, COMPAS ``--black-hole-kicks-mode REDUCED``), again with no kick for
  PPISN BHs; with ``keep_envelope`` the scaling uses the BH mass including the accreted envelope. "compas" uses the
  natal kick magnitudes applied by COMPAS itself. "auto" is
  "fallback" except for prescriptions that do not define a fallback fraction (MULLERMANDEL), where it is "compas";
  otherwise every Mueller-Mandel BH would get a full neutron-star kick.

Pair instability follows the prescription used to build the tables (HENDRIKS by default).
"""
import glob
import os
import warnings
from collections import namedtuple

import numpy as np
from scipy.special import erf

_TABLE_DIR = os.path.join(os.path.dirname(__file__), "data", "remnant_tables")

BH_TYPE = 14  # COMPAS stellar type for black holes
CCSN, PPISN = 1, 8  # COMPAS supernova types

bounds = namedtuple("bounds", ("lower", "upper"))
mbin = namedtuple("mbin", ("lower", "upper"))  # Same layout as ssptools.masses.mbin

KICK_MODELS = ("auto", "fallback", "reduced", "compas")
NS_MASS = 1.4  # [Msun] canonical NS mass for kick_model="reduced" (COMPAS NEUTRON_STAR_MASS)
# COMPAS prescriptions that do not define a fallback fraction (it is stored as 0 for every BH), so fallback-scaled
# kicks would give their BHs full neutron-star kicks. "auto" uses the kicks COMPAS applied for these.
NO_FALLBACK_PRESCRIPTIONS = ("MULLERMANDEL",)


def available_tables():
    """Names of the tables shipped with cBHBd."""
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(_TABLE_DIR, "*.npz")))


def is_table(ifmr):
    """True if ``ifmr`` names a shipped table or points to a table file."""
    return isinstance(ifmr, str) and (ifmr in available_tables() or ifmr.endswith(".npz"))


def _table_path(ifmr):
    return ifmr if ifmr.endswith(".npz") else os.path.join(_TABLE_DIR, f"{ifmr}.npz")


def table_mass_range(ifmr):
    """(min, max) ZAMS mass [Msun] covered by a table."""
    m = np.load(_table_path(ifmr))["m_zams"]
    return float(m[0]), float(m[-1])


def maxwellian_cdf(v, scale):
    """Cumulative distribution of a Maxwellian with 1D dispersion ``scale``, evaluated at speed ``v``."""
    scale = np.asarray(scale, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(v, dtype=float) / scale
        cdf = erf(x / np.sqrt(2)) - np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2)
    return np.where(scale > 0, cdf, 1.0)


class TabulatedRemnants:
    """Single-star remnant outcomes at one metallicity, read from a COMPAS table.

    Parameters
    ----------
    ifmr : str
        Table name (see ``available_tables()``) or path to a ``.npz`` table.
    FeH : float
        Metallicity [Fe/H]; converted to Z with ``Zsolar``. The nearest tabulated metallicity is used.
    Zsolar : float
        Solar metallicity used to convert [Fe/H] to Z. Must match the convention used by CBHBD (0.02 by default).
    keep_envelope : bool
        If True, core-collapse BHs also accrete the H envelope, scaled by the fallback fraction.
    kick_model : {"auto", "fallback", "reduced", "compas"}
        How natal kicks are computed, see the module documentation.
    """

    def __init__(self, ifmr, FeH, Zsolar=0.02, keep_envelope=False, kick_model="auto"):
        if kick_model not in KICK_MODELS:
            raise ValueError(f"Unknown kick_model {kick_model}, choose from {KICK_MODELS}")
        path = _table_path(ifmr)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Remnant table {ifmr} not found. Available tables: {available_tables()}")

        t = np.load(path)
        self.name = ifmr
        self.keep_envelope = keep_envelope
        self.compas_version = str(t["compas_version"])
        self.compas_command = str(t["compas_command"])
        no_fallback = any(f"--remnant-mass-prescription {p}" in self.compas_command
                          for p in NO_FALLBACK_PRESCRIPTIONS)
        if kick_model == "auto":
            kick_model = "compas" if no_fallback else "fallback"
        elif kick_model == "fallback" and no_fallback:
            warnings.warn(f"Table {ifmr} comes from a prescription without a fallback fraction (stored as 0), so "
                          "kick_model='fallback' gives every BH a full kick; kick_model='compas' or 'auto' is advised")
        self.kick_model = kick_model
        self.Z_grid = t["Z"]
        self.m_zams = t["m_zams"].astype(float)

        Z = Zsolar * 10 ** FeH
        if Z < self.Z_grid[0] * 0.999 or Z > self.Z_grid[-1] * 1.001:
            warnings.warn(f"Z = {Z:.3g} is outside the range of table {ifmr} ({self.Z_grid[0]:.3g} - "
                          f"{self.Z_grid[-1]:.3g}); using the nearest tabulated metallicity")
        self.iZ = int(np.argmin(np.abs(np.log10(self.Z_grid) - np.log10(Z))))
        self.Z = float(self.Z_grid[self.iZ])

        rem_type = t["rem_type"][self.iZ]
        m_rem = t["m_rem"][self.iZ].astype(float)
        m_he = t["m_he"][self.iZ].astype(float)
        m_tot = t["m_tot"][self.iZ].astype(float)
        fb = t["fb"][self.iZ].astype(float)
        sn_type = t["sn_type"][self.iZ]
        self.nseed = rem_type.shape[1]

        is_bh = rem_type == BH_TYPE
        mbh = np.where(is_bh, m_rem, 0.0)
        if keep_envelope:
            envelope = np.where(is_bh & (sn_type == CCSN), np.maximum(m_tot - m_he, 0.0), 0.0)
            mbh = mbh + fb * envelope
        self.mbh = mbh  # [Msun] BH mass for each (ZAMS mass, realisation); 0 if no BH forms

        # Factor scaling sigmans for each BH: (1 - fallback fraction) or NS_MASS / M_BH. PPISN BHs get no kick (as in
        # COMPAS by default).
        if kick_model == "reduced":
            kick_scale = np.minimum(NS_MASS / np.where(mbh > 0, mbh, np.inf), 1.0)
        else:
            kick_scale = 1 - fb
        self.kick_scale = np.where(sn_type == PPISN, 0.0, kick_scale)
        self.v_kick_compas = t["v_kick"][self.iZ].astype(float)

        if not is_bh.any():
            raise ValueError(f"Table {ifmr} forms no BHs at Z = {self.Z}")
        bh_rows = np.where(is_bh.any(axis=1))[0]
        self.BH_mi = bounds(self.m_zams[bh_rows[0]], self.m_zams[-1])  # [Msun] ZAMS range that can form BHs
        self.BH_mf = bounds(mbh[is_bh].min(), mbh[is_bh].max())  # [Msun] range of BH masses

        self._log_m0 = np.log(self.m_zams[0])
        self._dlog_m = np.log(self.m_zams[-1] / self.m_zams[0]) / (len(self.m_zams) - 1)

    def __repr__(self):
        return (f"TabulatedRemnants({self.name}, Z={self.Z:.3g}, keep_envelope={self.keep_envelope}, "
                f"kick_model={self.kick_model})")

    def _index(self, mi):
        # Nearest grid point in log ZAMS mass
        i = np.rint((np.log(mi) - self._log_m0) / self._dlog_m).astype(int)
        return np.clip(i, 0, len(self.m_zams) - 1)

    def retention_fraction(self, idx, seed, vesc, sigma):
        """Probability that the BH from grid point (idx, seed) is retained with escape velocity ``vesc``."""
        if self.kick_model == "compas":
            return (self.v_kick_compas[idx, seed] <= vesc).astype(float)
        return maxwellian_cdf(vesc, sigma * self.kick_scale[idx, seed])

    def sample(self, mi, sigma):
        """Draw BH masses and natal kicks [km/s] for stars of ZAMS mass ``mi``.

        Uses ``numpy.random`` so that runs are reproducible with ``CBHBD(seed=...)``. Stars that do not form a BH get
        a mass of zero.
        """
        mi = np.atleast_1d(mi)
        idx = self._index(mi)
        seed = np.random.randint(0, self.nseed, size=mi.size)
        mbh = self.mbh[idx, seed]
        if self.kick_model == "compas":
            vkick = self.v_kick_compas[idx, seed]
        else:
            scale = sigma * self.kick_scale[idx, seed]
            vkick = scale * np.linalg.norm(np.random.normal(size=(mi.size, 3)), axis=1)
        return mbh, vkick

    def bh_bins(self, m_breaks, nbins):
        """BH mass bins built like the default ssptools binning, but with finite edges."""
        nbins = np.atleast_1d(nbins)
        sides = np.concatenate([np.geomspace(m_breaks[i], m_breaks[i + 1], nbins[i] + 1)[(i > 0):]
                                for i in range(len(nbins))])
        lower, upper = sides[:-1], sides[1:]
        mask = upper > self.BH_mf.lower
        lower, upper = lower[mask].copy(), upper[mask].copy()
        lower[0] = self.BH_mf.lower * 0.999
        upper[-1] = max(upper[-1], self.BH_mf.upper * 1.001)
        return mbin(lower, upper)

    def initial_bh_population(self, m_breaks, a_slopes, nbins, N0, vesc, sigma, natal_kicks=True, npoints=50_000):
        """Build an ``ssptools.InitialBHPopulation`` from this table.

        Integrates the IMF over the BH-forming ZAMS range, averaging over the tabulated realisations and weighting
        each BH by its probability of surviving its natal kick. The returned object has the attributes used by
        ``cluster.Cluster`` (M, N, m, Mtot, Ntot, bins, IMF, age, Ns_lost, Ms_lost).
        """
        import ssptools
        from ssptools.ifmr import get_data

        IMF = ssptools.masses.PowerLawIMF(m_break=m_breaks, a=a_slopes, N0=N0, ext="zeros")
        mi_max = min(self.BH_mi.upper, m_breaks[-1])
        mi = np.linspace(self.BH_mi.lower, mi_max, npoints)
        dm = mi[1] - mi[0]
        weight = IMF.N(mi) * dm / self.nseed
        idx = self._index(mi)

        bins = self.bh_bins(m_breaks, nbins)
        edges = np.r_[bins.lower, bins.upper[-1]]
        M_BH = np.zeros(len(bins.lower))
        N_BH = np.zeros(len(bins.lower))
        for s in range(self.nseed):
            mf = self.mbh[idx, s]
            frem = self.retention_fraction(idx, s, vesc, sigma) if natal_kicks else np.ones_like(mf)
            ok = mf > 0
            inds = np.digitize(mf[ok], edges) - 1
            M_BH += np.bincount(inds, weights=(frem * mf * weight)[ok], minlength=len(M_BH))
            N_BH += np.bincount(inds, weights=(frem * weight)[ok], minlength=len(N_BH))

        out = ssptools.InitialBHPopulation(M_BH, N_BH, bins)
        out.IMF = IMF
        out.IFMR = self
        out.natal_kicks = natal_kicks

        # Same definitions as ssptools.InitialBHPopulation.from_IMF
        mstogrid = np.loadtxt(get_data("sevtables/msto.dat"))
        FeH = np.log10(self.Z / 0.02)
        a = mstogrid[np.argmin(np.abs(mstogrid[:, 0] - FeH)), 1:]
        out.age = a[0] * np.exp(a[1] * self.BH_mi.lower ** a[2])  # [Myr] time by which all BHs have formed
        out.Ns_lost = IMF.integrate_N(self.BH_mi.lower, mi_max)
        out.Ms_lost = IMF.integrate_M(self.BH_mi.lower, mi_max)
        return out
