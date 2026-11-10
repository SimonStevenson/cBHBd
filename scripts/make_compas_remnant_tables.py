"""
Build tabulated single-star remnant outcomes from COMPAS for use in cBHBd.

For each requested COMPAS remnant-mass prescription, this script evolves single stars (COMPAS ``--mode SSE``) on a
grid of ZAMS masses and metallicities, with several independent random seeds per grid point so that stochastic
prescriptions (e.g. the partial-fallback branch of MALTSEV2024, or MULLERMANDEL) are sampled. The outcomes are written
to ``cbhbd/data/remnant_tables/<name>.npz`` and read by :mod:`cbhbd.remnant_tables`.

The stored quantities per (metallicity, ZAMS mass, seed) are:

- ``rem_type``: COMPAS stellar type of the remnant (13 = NS, 14 = BH, 15 = massless remnant)
- ``m_rem``: COMPAS remnant mass [Msun]
- ``m_he``: He-core mass at CO-core formation [Msun]
- ``m_tot``: total stellar mass at CO-core formation [Msun] (used for the optional H-envelope fallback)
- ``fb``: COMPAS fallback fraction (used to scale natal kicks)
- ``sn_type``: COMPAS supernova type (1 = CCSN, 2 = ECSN, 4 = PISN, 8 = PPISN, ...)
- ``v_kick``: natal kick applied by COMPAS [km/s] (only used with ``kick_model="compas"`` in cBHBd)

Usage (needs a COMPAS executable and h5py)::

    python scripts/make_compas_remnant_tables.py --compas ~/Github/COMPAS/src/COMPAS --nproc 10

Variant tables, e.g. a Hendriks pair-instability CO-core shift of -5 Msun for Maltsev balanced (``--extra`` must come
last)::

    python scripts/make_compas_remnant_tables.py --prescriptions compas-maltsev-balanced --name-suffix=-ppi-5 \\
        --extra --PPI-CO-Core-Shift-Hendriks -5

Run ``python scripts/make_compas_remnant_tables.py --help`` for all options.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# name: (COMPAS options, stochastic?)
PRESCRIPTIONS = {
    "compas-maltsev-balanced": (["--remnant-mass-prescription", "MALTSEV2024", "--maltsev-mode", "BALANCED"], True),
    "compas-maltsev-optimistic": (["--remnant-mass-prescription", "MALTSEV2024", "--maltsev-mode", "OPTIMISTIC"],
                                  True),
    "compas-maltsev-pessimistic": (["--remnant-mass-prescription", "MALTSEV2024", "--maltsev-mode", "PESSIMISTIC"],
                                   True),
    "compas-fryer-delayed": (["--remnant-mass-prescription", "FRYER2012", "--fryer-supernova-engine", "DELAYED"], False),
    "compas-mullermandel": (["--remnant-mass-prescription", "MULLERMANDEL"], True),
}

# Options shared by every table, so that cluster and isolated-binary populations can use identical stellar physics.
COMMON_OPTIONS = ["--pulsational-pair-instability-prescription", "HENDRIKS"]


def _run_one(args):
    compas, name, Z, masses, nseed, extra, base_seed = args
    opts, stochastic = PRESCRIPTIONS[name]
    nseed = nseed if stochastic else 1
    workdir = tempfile.mkdtemp(prefix=f"cbhbd_{name}_")
    try:
        grid = os.path.join(workdir, "grid.txt")
        with open(grid, "w") as f:
            for m in masses:
                for _ in range(nseed):
                    f.write(f"--initial-mass {m:.6f} --metallicity {Z:.8g}\n")
        cmd = [compas, "--mode", "SSE", "--grid", grid, "--output-path", workdir, "--output-container", "out",
               "--random-seed", str(base_seed), "--quiet"] + COMMON_OPTIONS + opts + extra
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COMPAS failed for {name}, Z={Z}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        cmd_str = " ".join(["COMPAS"] + [c for c in cmd[1:] if not c.startswith(workdir)])
        return name, Z, _parse(os.path.join(workdir, "out", "COMPAS_Output.h5"), len(masses), nseed), cmd_str
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _parse(h5path, nm, nseed):
    import h5py

    with h5py.File(h5path, "r") as f:
        sp = f["SSE_System_Parameters"]
        seed = sp["SEED"][()]
        order = np.argsort(seed)  # grid lines are evolved with consecutive seeds
        rem_type = sp["Stellar_Type"][()][order]
        m_rem = sp["Mass"][()][order]
        n = len(seed)
        m_he = np.zeros(n)
        m_tot = np.zeros(n)
        fb = np.zeros(n)
        sn_type = np.zeros(n, dtype=int)
        v_kick = np.zeros(n)
        if "SSE_Supernovae" in f:
            sn = f["SSE_Supernovae"]
            idx = np.searchsorted(seed[order], sn["SEED"][()])
            m_he[idx] = sn["Mass_He_Core@CO"][()]
            m_tot[idx] = sn["Mass_Total@CO"][()]
            fb[idx] = sn["Fallback_Fraction"][()]
            sn_type[idx] = sn["SN_Type"][()]
            v_kick[idx] = sn["Applied_Kick_Magnitude"][()]
    if n != nm * nseed:
        raise RuntimeError(f"Expected {nm * nseed} stars in {h5path}, found {n}")
    shape = (nm, nseed)
    return dict(rem_type=rem_type.reshape(shape).astype(np.int8), m_rem=m_rem.reshape(shape).astype(np.float32),
                m_he=m_he.reshape(shape).astype(np.float32), m_tot=m_tot.reshape(shape).astype(np.float32),
                fb=fb.reshape(shape).astype(np.float32), sn_type=sn_type.reshape(shape).astype(np.int8),
                v_kick=v_kick.reshape(shape).astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compas", default=os.path.expanduser("~/Github/COMPAS/src/COMPAS"), help="COMPAS executable")
    p.add_argument("--prescriptions", nargs="+", default=list(PRESCRIPTIONS), choices=list(PRESCRIPTIONS))
    p.add_argument("--zmin", type=float, default=1e-4, help="Minimum metallicity (COMPAS lower limit is 1e-4)")
    p.add_argument("--zmax", type=float, default=0.03, help="Maximum metallicity (COMPAS upper limit is 0.03)")
    p.add_argument("--nz", type=int, default=23, help="Number of log-spaced metallicities")
    p.add_argument("--mmin", type=float, default=10.0, help="Minimum ZAMS mass [Msun]")
    p.add_argument("--mmax", type=float, default=150.0, help="Maximum ZAMS mass [Msun]")
    p.add_argument("--nm", type=int, default=1200, help="Number of log-spaced ZAMS masses")
    p.add_argument("--nseed", type=int, default=4, help="Random realisations per grid point (stochastic prescriptions)")
    p.add_argument("--seed", type=int, default=20261109, help="Base COMPAS random seed")
    p.add_argument("--nproc", type=int, default=os.cpu_count())
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "cbhbd", "data", "remnant_tables"))
    p.add_argument("--name-suffix", default="",
                   help="Appended to each table name, e.g. --name-suffix=-ppi-5 for a Hendriks CO-core shift of -5 passed in "
                        "--extra (use the = form: a value starting with '-' is otherwise read as an option)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra COMPAS options appended to every run (must come last)")
    a = p.parse_args()

    if shutil.which(a.compas) is None and not os.path.exists(a.compas):
        sys.exit(f"COMPAS executable not found: {a.compas}")
    version = next(line.strip() for line in subprocess.run([a.compas, "-v"], capture_output=True, text=True)
                   .stdout.splitlines() if line.strip())

    Zs = np.geomspace(a.zmin, a.zmax, a.nz)
    masses = np.geomspace(a.mmin, a.mmax, a.nm)
    tasks = [(a.compas, name, Z, masses, a.nseed, a.extra, a.seed + 1000 * i)
             for name in a.prescriptions for i, Z in enumerate(Zs)]
    results = {name: {} for name in a.prescriptions}
    commands = {}
    with ProcessPoolExecutor(max_workers=a.nproc) as ex:
        for k, (name, Z, out, cmd) in enumerate(ex.map(_run_one, tasks)):
            results[name][Z] = out
            commands.setdefault(name, cmd)
            print(f"[{k + 1}/{len(tasks)}] {name} Z={Z:.3g} done", flush=True)

    os.makedirs(a.outdir, exist_ok=True)
    for name in a.prescriptions:
        arrays = {key: np.stack([results[name][Z][key] for Z in Zs]) for key in results[name][Zs[0]]}
        path = os.path.join(a.outdir, f"{name}{a.name_suffix}.npz")
        np.savez_compressed(path, Z=Zs, m_zams=masses, compas_version=version, compas_command=commands[name],
                            **arrays)
        print(f"Wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
