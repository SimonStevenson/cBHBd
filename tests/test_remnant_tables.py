import numpy as np
import pytest

import cbhbd.cbhbd
from cbhbd import funcs, remnant_tables


def test_tables_available():
    tables = remnant_tables.available_tables()
    for name in ["compas-maltsev-balanced", "compas-maltsev-optimistic", "compas-maltsev-pessimistic",
                 "compas-fryer-delayed", "compas-mullermandel"]:
        assert name in tables, f"Missing remnant table {name}"


def test_maltsev_gap_in_1G_masses():
    # MALTSEV2024 with single stars leaves no 1G BHs between roughly 10 and 16 Msun at these metallicities
    for FeH in (np.log10(0.0006 / 0.02), np.log10(0.006 / 0.02)):
        table = remnant_tables.TabulatedRemnants("compas-maltsev-balanced", FeH)
        mbh = table.mbh[table.mbh > 0]
        assert np.mean((mbh > 10.5) & (mbh < 15.5)) == 0, "Unexpected BHs inside the Maltsev gap"


def test_neutron_star_islands_are_kept():
    # Above the lowest BH-forming mass, some stars still make neutron stars (islands of explodability)
    table = remnant_tables.TabulatedRemnants("compas-maltsev-balanced", np.log10(0.006 / 0.02))
    above = table.m_zams >= table.BH_mi.lower
    assert np.any(table.mbh[above] == 0), "No neutron-star islands found above the BH threshold"


def test_cluster_and_sampled_populations_agree():
    model = cbhbd.cbhbd.CBHBD(M0=3e5, rh0=1, Z=0.001, compute_mergers=False, seed=3, verbose=False,
                              ifmr="compas-maltsev-balanced")
    cl = model.cluster
    np.random.seed(11)
    v_esc0 = 50 * (model.M0 / 1e5) ** (1 / 3) * (model.rhoh0 / 1e5) ** (1 / 6)
    _, bhv = funcs.sample_BHs(cl.Mbh[0], v_esc0, model.m_breaks[-2], model.mmax, model.a_slopes[-1], model.FeH,
                              model.ifmr, cl.sigmans, model.mmax_IMF, table=cl.remnant_table)
    assert bhv[:, 0].sum() == pytest.approx(cl.Mbh0, rel=0.05)
    assert bhv[:, 0].mean() == pytest.approx(cl.mbh0, rel=0.1)


def test_keep_envelope_increases_bh_masses():
    FeH = np.log10(0.006 / 0.02)
    lean = remnant_tables.TabulatedRemnants("compas-maltsev-balanced", FeH)
    fat = remnant_tables.TabulatedRemnants("compas-maltsev-balanced", FeH, keep_envelope=True)
    assert np.all(fat.mbh >= lean.mbh)
    assert fat.mbh.sum() > lean.mbh.sum()


def test_mergers_with_table_are_reproducible():
    kw = dict(M0=3e5, rh0=1, Z=0.002, seed=7, verbose=False, ifmr="compas-maltsev-balanced")
    a = cbhbd.cbhbd.CBHBD(**kw).mergers
    b = cbhbd.cbhbd.CBHBD(**kw).mergers
    assert len(a) > 0
    assert a.equals(b)
    assert (a.gen == a[["gen1", "gen2"]].max(axis=1)).all()


def test_abh_not_supported_with_tables():
    with pytest.raises(NotImplementedError):
        cbhbd.cbhbd.CBHBD(M0=1e5, rh0=1, Z=0.002, ifmr="compas-maltsev-balanced", add_aBH=True, verbose=False)


def test_auto_kicks_for_prescriptions_without_fallback():
    FeH = np.log10(0.002 / 0.02)
    mm = remnant_tables.TabulatedRemnants("compas-mullermandel", FeH)
    assert mm.kick_model == "compas"
    assert remnant_tables.TabulatedRemnants("compas-maltsev-balanced", FeH).kick_model == "fallback"
    bh = np.where(mm.mbh > 0)
    assert np.mean(mm.retention_fraction(*bh, 111.0, 217.0)) > 0.5
    with pytest.warns(UserWarning):
        remnant_tables.TabulatedRemnants("compas-mullermandel", FeH, kick_model="fallback")
