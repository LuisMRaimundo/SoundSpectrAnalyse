from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)

# Ensure repository modules always win import resolution over unrelated
# machine-local files that may share module names.
sys.path = [p for p in sys.path if p != project_root_str]
sys.path.insert(0, project_root_str)


def _pin_repo_module(module_name: str, file_name: str) -> None:
    module_path = PROJECT_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_pin_repo_module("acoustic_density_core", "acoustic_density_core.py")
