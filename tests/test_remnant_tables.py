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


def test_reduced_kicks():
    # Momentum-conserving kicks (COMPAS --black-hole-kicks-mode REDUCED): sigma scaled by 1.4 Msun / M_BH, no kick for
    # PPISN BHs
    FeH = np.log10(0.002 / 0.02)
    red = remnant_tables.TabulatedRemnants("compas-maltsev-balanced", FeH, kick_model="reduced")
    fb = remnant_tables.TabulatedRemnants("compas-maltsev-balanced", FeH)
    bh = red.mbh > 0
    ppisn = np.load(remnant_tables._table_path("compas-maltsev-balanced"))["sn_type"][red.iZ] == remnant_tables.PPISN
    assert np.allclose(red.kick_scale[bh & ~ppisn], remnant_tables.NS_MASS / red.mbh[bh & ~ppisn])
    assert (red.kick_scale[bh & ppisn] == 0).all()
    np.random.seed(1)
    m, v = red.sample(np.random.uniform(20, 60, 40_000), 217.0)
    ok = (m > 0) & (m < 30)  # core-collapse BHs only (PPISN BHs are heavier)
    assert ok.sum() > 5_000
    sigma = 217.0 * remnant_tables.NS_MASS / m[ok]
    assert abs(np.mean((v[ok] / sigma) ** 2) - 3) < 0.1  # Maxwellian: <v^2> = 3 sigma^2
    # Fallback kicks leave most Maltsev BHs unkicked, so reduced kicks retain fewer at low escape speed
    idx = np.where(bh)
    assert np.mean(red.retention_fraction(*idx, 20.0, 217.0)) < np.mean(fb.retention_fraction(*idx, 20.0, 217.0))
