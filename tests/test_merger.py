import cbhbd.cbhbd
import numpy as np


def test_merger():
    print("Checking that the code runs without crashing when mergers are enabled", flush=True)
    tend = 13e3
    M0 = 8e5
    rg = 8
    rh0 = 1
    Z = 0.003
    seed = 12345
    verbose = True

    model = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=True, rg=rg, seed=seed, verbose=verbose)

    assert model.cluster is not None, "Cluster is not properly computed"
    assert model.mergers is not None, "Mergers are not properly computed"
    assert len(model.mergers) != 0, "No mergers found but some are expected for these initial conditions"

    assert model.mIMBH > 0, f"IMBH mass is not properly computed, {model.mIMBH = }"
    assert model.genIMBH >= 1, f"IMBH generation is not properly computed, {model.genIMBH = }"
    # 1G BHs have zero natal spin. If every merger product was ejected, the most massive retained BH is 1G and
    # spinless, which is a legitimate outcome for some seeds.
    assert model.chiIMBH >= 0, f"IMBH spin is not properly computed, {model.chiIMBH = }"
    if model.genIMBH > 1:
        assert model.chiIMBH > 0, f"Merger product with zero spin, {model.chiIMBH = }"

    assert model.mIMBHej > 0, f"IMBHej mass is not properly computed, {model.mIMBHej = }"
    assert model.chiIMBHej > 0, f"IMBHej spin is not properly computed, {model.chiIMBHej = }"
    assert model.genIMBHej > 0, f"IMBHej generation is not properly computed, {model.genIMBHej = }"

    assert {"gen1", "gen2"} <= set(model.mergers.columns), "Per-component generations missing from the output"
    assert (model.mergers.gen == model.mergers[["gen1", "gen2"]].max(axis=1)).all()

    assert model.cluster.M_interp(model.cluster.tfin) >= 0, "Final cluster mass is not properly computed"
    assert model.cluster.Mbh_interp(model.cluster.tfin) >= 0, "Final BH mass is not properly computed"
    assert model.cluster.rh_interp(model.cluster.tfin) >= 0, "Final half-mass radius is not properly computed"


def test_merger_small_cluster():
    print(
        "Checking that the code runs without crashing when mergers are enabled and the cluster is too small to have BHs",
        flush=True)
    tend = 13e3
    M0 = 1e2
    rg = 8
    rh0 = 1
    Z = 0.003
    seed = 12345
    verbose = True

    model = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=True, rg=rg, seed=seed, verbose=verbose)


def test_merger_delay_is_coalescence_time():
    # t_merge - t_sim is the GW coalescence time of the output (a, e) (Mandel 2021 fit), not a/|da/dt|
    import numpy as np
    from cbhbd import mergers
    from cbhbd.cbhbd import CBHBD

    m = CBHBD(M0=3e5, rh0=1, Z=0.002, seed=3, verbose=False).mergers
    ej = m[m.merger_type == "ejected"]
    assert len(ej) > 0
    G, c = 4 * np.pi ** 2, 63198
    T = mergers.coalescence_time(ej.a, ej.e, ej.m1, ej.m2, G, c)
    # t_merge = t + T, with t the synchronised time (>= t_sim)
    assert np.all(ej.t_merge - T >= ej.t_sim * (1 - 1e-9))
    R = 1 + 73 / 24 * ej.e ** 2 + 37 / 96 * ej.e ** 4
    tau = 5 * c ** 5 * ej.a ** 4 * (1 - ej.e ** 2) ** 3.5 / (64 * G ** 3 * ej.m1 * ej.m2 * (ej.m1 + ej.m2) * R)
    assert np.any(ej.t_merge - tau < ej.t_sim * (1 - 1e-6))  # the old delay a/|da/dt| is not what was added
    assert mergers.T_MERGE_DEFINITION == "coalescence_mandel2021"
