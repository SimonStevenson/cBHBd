# ᴄBHBᴅ

A fast code for simulating star clusters and black hole dynamics.

ᴄBHBᴅ can be used for

1) the evolution of a star cluster with black holes in a tidal field
2) population synthesis of binary black hole mergers in a time-evolving star cluster

If you use ᴄBHBᴅ in any publication, please cite it as follows:
> ᴄBHBᴅ (Antonini & Gieles [2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2936A/abstract); Antonini et
> al. [2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522..466A/abstract)), with the updates of Fronimos Pouliasis et
> al. ([2025](https://ui.adsabs.harvard.edu/abs/2026arXiv260528088F/abstract)),
> Chattopadhyay et al. ([2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260409773C/abstract)), and Marín Pina et al.
> (in prep)

ᴄBHBᴅ stands for the combination of ᴄʟᴜsᴛᴇʀBH (the code for the cluster evolution) and BHBᴅʏɴᴀᴍɪᴄs (the code for the
black hole mergers).

## Installation

The cBHBd package can be installed using pip:

```
pip install cbhbd
```

All required dependencies will be installed automatically.

You can also simply download the ᴄBHBᴅ package source from this repository and install or place it into your path:

```
git clone https://github.com/cBHBd/cBHBd
```

## Running ᴄBHBᴅ

You can run ᴄBHBᴅ as

```
from cbhbd.cbhbd import CBHBD

c = CBHBD(...)
```

with the following arguments:

- `N`: Initial number of stars _or_ `M0`: Initial cluster mass
- `rhoh0`: Initial half-mass density [Msun/pc^3] _or_ `rh0`: Initial half-mass radius [pc]
- `Z`: Metallicity _or_ `FeH`: Metallicity as in [Fe/H]
- `kwargs`: Additional parameters to override defaults (see code comments in the `CBHBD` class and the `Cluster` class)

If you only want to run the cluster evolution, you can use `compute_mergers=False` for efficiency

### Remnant masses from COMPAS

First-generation BH masses can be taken from tables of COMPAS single-star outcomes instead of the ssptools IFMRs, so
that cluster and isolated-binary models share the same stellar physics:

```
c = CBHBD(M0=1e6, rh0=1, Z=0.001, ifmr="compas-maltsev-balanced")
```

Available tables (`cbhbd.remnant_tables.available_tables()`):

- `compas-maltsev-balanced`, `compas-maltsev-optimistic`, `compas-maltsev-pessimistic`: the Maltsev et al. (2025)
  compactness-based prescription (`MALTSEV2024` in COMPAS), with the three low-metallicity variants of Willcox et al.
  (2025)
- `compas-fryer-delayed`: Fryer et al. (2012) delayed
- `compas-mullermandel`: Mandel & Müller (2020), the COMPAS default
- `compas-maltsev-balanced-ppi-10`, `-ppi-5`, `-ppi+5`: Maltsev balanced with the HENDRIKS pulsational
  pair-instability CO-core shift (`--PPI-CO-Core-Shift-Hendriks`) set to -10, -5 and +5 Msun; at Z = 2.2e-4 the
  maximum 1G BH mass is 39.5, 45.1 and 56.2 Msun, against 50.7 Msun with no shift. Built on OzSTAR with the same
  COMPAS commit; an unshifted rebuild there matched the laptop table to about 1e-6 relative (compiler and library
  differences), with no change in any remnant or supernova type.
- `compas-maltsev-balanced-brcek`: Maltsev balanced with BRCEK main-sequence cores (Shikauchi et al. 2024,
  `--main-sequence-core-mass-prescription BRCEK`; the default is MANDEL, and HURLEY gives a table identical to
  MANDEL). BRCEK changes single stars: at Z = 0.011 the BH fraction (10-150 Msun ZAMS) rises from 0.10 to 0.24 and
  the median BH mass from 7.8 to 21.5 Msun, and at Z = 1e-3 the maximum BH mass from 44 to 50 Msun. No BH lands in
  12-16 Msun at any metallicity. Built on OzSTAR with the same COMPAS commit.

All tables use the `HENDRIKS` pulsational pair-instability prescription and cover Z = 1e-4 to 0.03 and ZAMS masses of
10 to 150 Msun. Each grid point stores several COMPAS realisations, so stochastic prescriptions are sampled, and
neutron-star "islands" above the lowest BH-forming mass are kept. Related options:

- `keep_envelope` (default `False`): if `True`, core-collapse BHs also accrete the hydrogen envelope, scaled by the
  fallback fraction. COMPAS `MALTSEV2024` sets the BH mass to the He-core mass.
- `kick_model` (default `"auto"`): `"fallback"` draws Maxwellian natal kicks with dispersion `sigmans`, scaled by
  (1 - fallback fraction), with no kick for BHs from pulsational pair instability. `"reduced"` scales `sigmans` by
  1.4 Msun / M_BH instead (momentum-conserving kicks, COMPAS `--black-hole-kicks-mode REDUCED`), again with no kick
  for pulsational pair-instability BHs. `"compas"` uses the kicks applied
  by COMPAS. `"auto"` is `"fallback"` except for `compas-mullermandel`, whose prescription defines no fallback
  fraction, where it uses the COMPAS kicks.

The default GW recoil model (`remnant_model="RemnantModelSimple"`) computes remnant spins and kicks but sets the
remnant mass to m1 + m2. The NR-surrogate models `RemnantModelVarma19Islam23` and `RemnantModelIslam26` include the
mass radiated in GWs; install their dependencies with `pip install ".[nr]"` (surfinBH downloads its fit data, a few
minutes, on first use).

Tables are built with `scripts/make_compas_remnant_tables.py`, which needs a COMPAS executable and `h5py`. The aBH
option is not available with these tables.

## Analysing the ᴄBHBᴅ output

The output of the simulations is stored in a `CBHBD` object. The cluster properties can be accessed from `c.cluster`.
See `cluster.py` for the full documentation.

```
c.cluster.Mst # Total stellar mass in the cluster as function of time
c.cluster.Mbh # Total BH mass in the cluster as function of time
c.cluster.rh # Half-mass radius of the cluster as function of time
...
c.cluster.Mst_interp(t) # For convenience, you can also interpolate the cluster properties at any time using the interpolating functions
```

The merger properties are stored in a `pandas.Dataframe` in `c.mergers`. See `mergers.py` for the full documentation.

```
c.mergers # The merger data, use print(c.mergers) to see a table of the merger properties
c.mergers.m1 # The mass of the primary
c.mergers.merger_type # The type of merger (ejected, gw_capture, ...), see the code for the full documentation
c.mergers.gen1, c.mergers.gen2 # Generation of each component (1 = first generation); c.mergers.gen is the larger
... 
c.mergers.to_numpy() # You can also export the merger data to a numpy array
```

