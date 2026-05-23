from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd

import dissonance_models as dm


def _legacy_total_and_pairs(model: dm.DissonanceModel, df: pd.DataFrame) -> tuple[float, int]:
    dfx = df.copy()
    if "Amplitude" not in dfx.columns:
        dfx["Amplitude"] = 10 ** (dfx["Magnitude (dB)"] / 20)
    dfx = dfx[dfx["Frequency (Hz)"] > 0]
    freqs = dfx["Frequency (Hz)"].to_numpy(dtype=float)
    amps = dfx["Amplitude"].to_numpy(dtype=float)
    total = 0.0
    n_pairs = 0
    for i in range(len(freqs) - 1):
        for j in range(i + 1, len(freqs)):
            try:
                total += model.pure_tones_dissonance(freqs[i], freqs[j], amps[i], amps[j])
                n_pairs += 1
            except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
                continue
    return float(total), int(n_pairs)


def _legacy_total_pairs_and_minamp(model: dm.DissonanceModel, df: pd.DataFrame) -> tuple[float, int, float]:
    dfx = df.copy()
    if "Amplitude" not in dfx.columns:
        dfx["Amplitude"] = 10 ** (dfx["Magnitude (dB)"] / 20)
    dfx = dfx[(dfx["Frequency (Hz)"] > 0) & (dfx["Amplitude"] > 0)]
    freqs = dfx["Frequency (Hz)"].to_numpy(dtype=float)
    amps = dfx["Amplitude"].to_numpy(dtype=float)
    total = 0.0
    n_pairs = 0
    sum_minamp = 0.0
    for i in range(len(freqs) - 1):
        for j in range(i + 1, len(freqs)):
            sum_minamp += min(amps[i], amps[j])
            total += model.pure_tones_dissonance(freqs[i], freqs[j], amps[i], amps[j])
            n_pairs += 1
    return float(total), int(n_pairs), float(sum_minamp)


def _legacy_sethares_pairwise_sum(model: dm.SetharesDissonance, partials: list[tuple[float, float]]) -> float:
    ps = [(float(f), float(a)) for f, a in partials if f > 0 and a > 0]
    if len(ps) < 2:
        return 0.0
    ps.sort(key=lambda x: x[0])
    total = 0.0
    for i in range(len(ps) - 1):
        for j in range(i + 1, len(ps)):
            f1, a1 = ps[i]
            f2, a2 = ps[j]
            total += model.pure_tones_dissonance(f1, f2, a1, a2)
    return float(total)


def test_vectorised_pairwise_paths_match_legacy_loops() -> None:
    rng = np.random.default_rng(12345)
    freqs = np.sort(rng.uniform(80.0, 5000.0, size=64))
    amps = rng.uniform(1e-4, 1.0, size=64)
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})
    model = dm.SetharesDissonance()

    legacy_total, legacy_pairs = _legacy_total_and_pairs(model, df)
    new_total, new_pairs = model._dissonance_total_and_pairs(df)
    npt.assert_allclose(new_total, legacy_total, rtol=1e-10, atol=1e-12)
    assert new_pairs == legacy_pairs

    legacy_total2, legacy_pairs2, legacy_minamp = _legacy_total_pairs_and_minamp(model, df)
    new_total2, new_pairs2, new_minamp = model._dissonance_total_pairs_and_minamp(df)
    npt.assert_allclose(new_total2, legacy_total2, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(new_minamp, legacy_minamp, rtol=1e-10, atol=1e-12)
    assert new_pairs2 == legacy_pairs2

    partials = list(zip(freqs.tolist(), amps.tolist()))
    legacy_pw = _legacy_sethares_pairwise_sum(model, partials)
    new_pw = model._pairwise_sum(partials)
    npt.assert_allclose(new_pw, legacy_pw, rtol=1e-10, atol=1e-12)
