# Formula Extraction Table — Pass 2 — Weight Functions

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `WeightFunction.linear` | `return x` | \(w(x)=x\) | \(x\): input scalar/array | |
| `WeightFunction.squared` | `np.square(x)` | \(w(x)=x^2\) | | |
| `WeightFunction.sqrt` | `np.sqrt(x)` | \(w(x)=\sqrt{x}\) | | |
| `WeightFunction.cbrt` | `np.sign(x) * (np.abs(x) ** (1.0 / 3.0))` | \(w(x)=\operatorname{sign}(x)\,|x|^{1/3}\) | | |
| `WeightFunction.cubic` | `x ** 3` | \(w(x)=x^3\) | | |
| `WeightFunction.logarithmic` | `np.log1p(x)` | \(w(x)=\ln(1+x)\) | | |
| `WeightFunction.exponential` | `np.expm1(x)` | \(w(x)=e^x-1\) | | |
| `WeightFunction.inverse_log` | `1.0 / (np.log1p(x) + eps)` | \(w(x)=1/(\ln(1+x)+\varepsilon)\) | \(\varepsilon=10^{-10}\) | |
| `get_weight_function` | `key = (name or '').strip().lower()` | Normalised lookup key | | |
| `get_weight_function` | `if key == "sum": key = "linear"` | Alias: `sum` \(\mapsto\) `linear` | | |
| `get_weight_function` | `if key == "d2": key = "linear"` | Alias: `d2` \(\mapsto\) `linear` | | |
| `get_weight_function` | `if key == "d8": key = "d17"` | Alias: `d8` \(\mapsto\) `d17` | | |
| `get_weight_function` | registry maps discrete keys to callables | `d3`,`d10`,`d24` \(\to\) `logarithmic`; `d17` \(\to\) `linear` | | Discrete keys also short-circuit elsewhere |
| `DEFAULT_HARMONIC_ROLLOFF_ALPHA` | literal `1.5` | \(\alpha_{\mathrm{default}}=1.5\) | | Constant |
| `DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION` | literal `"logarithmic"` | — | | String default only |
