import numpy as np
from density import calculate_harmonic_density, calculate_inharmonic_density

def test_calculate_inharmonic_density_passes_expected_count_keyword():
    amps = np.ones(10, dtype=float)

    expected = calculate_harmonic_density(
        amps,
        threshold_db=-60.0,
        fundamental_freq=None,
        sr=None,
        include_amp_factor=True,
        amp_weight=0.20,
        max_expected_harmonics=50,
    )

    got = calculate_inharmonic_density(amps, threshold_db=-60.0, max_expected_partials=50)

    assert got == expected

