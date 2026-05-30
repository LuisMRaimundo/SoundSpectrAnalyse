# Metric Formula Index

| Formula ID | Formula | Metric | Code function | Export column |
|---|---|---|---|---|
| F-001 | $X(k,t)=\sum_{n=0}^{N-1}x[n+tH]w[n]e^{-j2\pi kn/N}$ | STFT | `proc_audio.py` STFT path (`fft_analysis`) | upstream to many metrics |
| F-002 | $A(k,t)=|X(k,t)|$ | Magnitude spectrum | STFT path | basis for amplitude metrics |
| F-003 | $P(k,t)=|X(k,t)|^2$ | Power spectrum | STFT path | basis for power/energy metrics |
| F-004 | $f_k=\frac{k f_s}{N}$ | FFT-bin frequency | STFT path | bin-frequency-dependent metrics |
| F-005 | $\Delta c = 1200\log_2(f_c/f_n)$ | Cents deviation | `acoustic_density_core.py`, `inharmonicity_model.py` | `f0_deviation_cents`, fit residuals |
| F-006 | $f_n=nf_0$ | Ideal harmonic prediction | harmonic matching logic | harmonic candidate/order columns |
| F-007 | $f_n=nf_0\sqrt{1+Bn^2}$ | Stiff-string prediction | `inharmonicity_model.fit_inharmonicity_coefficient` | inharmonicity diagnostics |
| F-008 | $y_n=(f_n/(nf_0))^2-1\approx Bn^2$ | Linearized B fit relation | `inharmonicity_model.fit_inharmonicity_coefficient` | `inharmonicity_coefficient_B` |
| F-009 | $\tau_n=\max(\tau_{\text{cents}},1200\Delta f_{\text{bin}}/(nf_0))$ | Adaptive harmonic tolerance | policy in `constants.py` + usage path | harmonic alignment and matching diagnostics |
| F-010 | $H=-\sum_i p_i\log_2 p_i$ | Shannon entropy | `_normalized_entropy` | intermediate entropy |
| F-011 | $H_{norm}=H/\log_2 K$ | Normalized entropy | `_normalized_entropy` | `spectral_entropy` |
| F-012 | $D_{eff}=(\sum_i P_i)^2/\sum_i P_i^2$ | Effective component count/density | `_effective_count`, density helpers | `effective_partial_density`, related fields |
| F-013 | $D_H=\sum_{i\in H}\phi(A_i)$ | Harmonic density sum | compile extraction + weight function path | `harmonic_density_sum` |
| F-014 | $D_I=\sum_{i\in I}\phi(A_i)$ | Inharmonic density sum | compile extraction + weight function path | `inharmonic_density_sum` |
| F-015 | $D_S=\sum_{i\in S}\phi(A_i)$ | Subbass density sum | compile extraction + weight function path | `subbass_density_sum` |
| F-016 | $D_{raw}=w_HD_H+w_ID_I+w_SD_S$ | Canonical weighted density | compile weighted composition | `density_metric_raw` |
| F-017 | $D_{per-note}=r_HD_H+r_ID_I+r_SD_S$ | Per-note balance density | compile weighted composition | `density_metric_raw_per_note_balance` |
| F-018 | $r_H=E_H/(E_H+E_I+E_S)$ (analogous $r_I,r_S$) | Per-note energy ratios | per-note extraction/metadata | `component_*_energy_ratio` |
| F-019 | $D_{norm}=D_{raw}/\max(D_{raw}^+)$ | Corpus-relative normalization | compile normalization pass | `density_metric_normalized` |
| F-020 | $f_{sub,max}=\min(0.5f_0,80)$ | Subbass upper bound | `SubBassPolicy.upper_bound_hz` | subbass policy columns |
| F-021 | $\alpha_{peak\_amp}(N)=N_{ref}/N$ | Tier normalization for peak-amplitude sums (`quantity_kind="peak_amplitude_sum"`) | `spectral_normalization.n_fft_normalization_factor` | `*_amplitude_sum_tier_normalized` |
| F-022 | $\alpha_{peak\_pow}(N)=(N_{ref}/N)^2$ | Tier normalization for peak-power sums (`quantity_kind="peak_power_sum"`) | `spectral_normalization.n_fft_normalization_factor` | `*_energy_sum_tier_normalized` |
| F-023 | $o_H=s_H/(s_H+s_I+s_S)$ (analogous $o_I,o_S$) | Pure observation triplet | acoustic core/adaptive path | `pure_observation_w_*` |
| F-024 | $\alpha\leftarrow (1-\lambda)\alpha + g\,o$ | Adaptive concentration update | `AdaptiveDensityEngine.update` | adaptive state/profile exports |
| F-025 | $\text{JSD}(p,q)=\frac12(D_{KL}(p\|m)+D_{KL}(q\|m))$ | Divergence gate | `adaptive_density_engine._js_divergence` | `js_divergence`, reliability |
| F-026 | $\text{rel}\propto e^{-\text{JSD}/T}$ (clipped) | Reliability attenuation | `AdaptiveDensityEngine.update` | `reliability` |
| F-027 | $C=\sum_i f_i p_i$ | Spectral centroid | `mir_descriptors.compute_mir_descriptors_from_spectrum` | `spectral_centroid_hz` |
| F-028 | $\sigma=\sqrt{\sum_i(f_i-C)^2p_i}$ | Spectral spread | same | `spectral_spread_hz` |
| F-029 | $\sum_i((f_i-C)/\sigma)^3p_i$ | Spectral skewness | same | `spectral_skewness` |
| F-030 | $\sum_i((f_i-C)/\sigma)^4p_i$ | Spectral kurtosis | same | `spectral_kurtosis` |
| F-031 | $\sum_i|A_{i+1}-A_i|/\sum_iA_i$ | Spectral irregularity | same | `spectral_irregularity` |
| F-032 | $T_1=A_1/\sum_i A_i$ | Tristimulus 1 | same | `tristimulus_1_fundamental` |
| F-033 | $T_2=(A_2+A_3+A_4)/\sum_iA_i$ | Tristimulus 2 | same | `tristimulus_2_low_harmonics_2_to_4` |
| F-034 | $T_3=\sum_{i\ge5}A_i/\sum_iA_i$ | Tristimulus 3 | same | `tristimulus_3_high_harmonics_5_plus` |
| F-035 | $F=\exp(\frac1K\sum_i\ln P_i)/(\frac1K\sum_iP_i)$ | Spectral flatness | same | `spectral_flatness` |
| F-036 | $\sum_{f_i\le R_p}P_i=p\sum_iP_i$ | Spectral rolloff | same | `spectral_rolloff_hz_85`, `spectral_rolloff_hz_95` |
| F-037 | $x=\frac{|f_i-f_j|}{0.25\min(f_i,f_j)+24.7},\ g(x)=xe^{1-x}$ | Aures-like roughness kernel | `_roughness_aures_1985` | `roughness_aures_1985` |
| F-038 | $\text{ERB}(f)=21.4\log_{10}(1+0.00437f)$ | ERB-rate transform | `_erb_rate_hz` | ERB grouping intermediary |
| F-039 | $D_{ERB}=1/\sum_b q_b^2$ | ERB effective density | MIR ERB grouping logic | `erb_weighted_spectral_density` |
| F-040 | $\mathrm{LAT}=\log_{10}(t_{attack})$ | Log attack time | `temporal_segmentation.segment_attack_sustain_release` | `log_attack_time_s` |
| F-041 | $\text{SBTI}=0.45z(BWED)+0.25z(LMER)+0.20z(HBDN)+0.10z(RBCC)$ | Research workbook body-thickness index | `build_spectral_density_metrics` | `spectral_body_thickness_index` |
| F-042 | $D_{final}=r_HD_H+r_ID_I+r_SD_S$ (with $r_*$ = measured `component_*_energy_ratio`, $D_*$ = `*_density_sum`) | Principled per-note scalar density (measured component balance × GUI-weighted per-band sums) | `compile_metrics._compute_note_density_final` | `note_density_final` |
| F-043 | $T=\alpha\,\hat{\mu}_{noise}$, $\alpha=N(P_{fa}^{-1/N}-1)$; detect iff $P_{peak}\ge T$ | Cell-averaging CFAR harmonic acceptance gate (adaptive, stated false-alarm rate; replaces fixed 3 dB SNR) | `harmonic_peak_validation.cfar_peak_detection` | `cfar_margin_db`, `cfar_detected` |
| F-044 | $\text{CI}_{95\%}$ via non-parametric bootstrap of per-partial contributions with ratios recomputed per resample | Uncertainty of `note_density_final` (partials + ratios propagated jointly) | `density_uncertainty.bootstrap_note_density_final` | `note_density_final_ci_low/ci_high/rel_uncertainty/uncertainty_sources` |
| F-045 | $N_{eff}=(\sum_n A_n^2)^2/\sum_n A_n^4$ | Energy-distribution density ("fatness"): effective number of harmonic partials carrying energy (register-robust, timbre-discriminating) | `compile_metrics._energy_distribution_density` | `harmonic_effective_partial_count` |
| F-046 | $E_{>f_0}=1-A_1^2/\sum_n A_n^2$ ; $\bar{n}=\sum_n nA_n^2/\sum_n A_n^2$ | Harmonic energy above fundamental & energy centroid order (spectral spread / brightness) | `compile_metrics._energy_distribution_density` | `harmonic_energy_above_fundamental_ratio`, `harmonic_energy_centroid_order` |
