# References

This file is the canonical bibliography for all theoretical anchors used in the SoundSpectrAnalyse codebase. Inline `References` blocks in individual modules use short form; full APA-7 entries live here. The dissertation that consumes this software should cite from this file.

## Spectral analysis and DFT/window theory
- Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. *Proceedings of the IEEE, 66*(1), 51–83.
- Heinzel, G., Rüdiger, A., & Schilling, R. (2002). *Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new at-top windows* (Technical report). Max-Planck-Institut für Gravitationsphysik.

## Inharmonicity and string acoustics
- Fletcher, H. (1962). Normal vibration frequencies of a stiff piano string. *Journal of the Acoustical Society of America, 36*(1), 203–209.
- Fletcher, N. H., & Rossing, T. D. (1998). *The physics of musical instruments* (2nd ed.). Springer.

## Psychoacoustics
- Zwicker, E., & Fastl, H. (1990). *Psychoacoustics: Facts and models*. Springer.
- Moore, B. C. J., & Glasberg, B. R. (1983). Suggested formulae for calculating auditory-filter bandwidths and excitation patterns. *Journal of the Acoustical Society of America, 74*(3), 750–753.
- Aures, W. (1985). Ein Berechnungsverfahren der Rauhigkeit. *Acustica, 58*(5), 268–281.

## Timbre and MIR descriptors
- Pollard, H. F., & Jansson, E. V. (1982). A tristimulus method for the specification of musical timbre. *Acustica, 51*(3), 162–171.
- Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011). The Timbre Toolbox: Extracting audio descriptors from musical signals. *Journal of the Acoustical Society of America, 130*(5), 2902–2916.

## Bayesian inference and information theory
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory, 37*(1), 145–151.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian data analysis* (3rd ed.). CRC Press.

## Scientific software methodology
- Hatton, L. (1997). The T-experiments: Errors in scientific software. *IEEE Computational Science and Engineering, 4*(2), 27–38.
- Soergel, D. A. W. (2015). Rampant software errors may undermine scientific results. *F1000Research, 3*, 303.

## Module-to-reference mapping

| Module | References used |
|---|---|
| `spectral_normalization.py` | Harris (1978); Heinzel et al. (2002) |
| `inharmonicity_model.py` | Fletcher (1962); Fletcher & Rossing (1998) |
| `subbass_policy.py` | Zwicker & Fastl (1990) |
| `mir_descriptors.py` | Moore & Glasberg (1983); Aures (1985); Pollard & Jansson (1982); Peeters et al. (2011) |
| `adaptive_density_engine.py` | Lin (1991); Gelman et al. (2013) |
| `metric_contract.py` | Hatton (1997); Soergel (2015) |
| `temporal_segmentation.py` | Peeters et al. (2011) |
