"""
Smoke imports for editable-install / console entry points.

Does not start the GUI or run analysis — only verifies modules resolve on sys.path
the same way pip install -e . exposes them via py-modules.
"""


def test_entrypoint_modules_import() -> None:
    import compile_metrics  # noqa: F401
    import density  # noqa: F401
    import harmonic_alignment  # noqa: F401
    import harmonic_validation  # noqa: F401
    import gui_model_weight_policy  # noqa: F401
    import interface  # noqa: F401
    import main  # noqa: F401
    import peak_component_counts  # noqa: F401
    import proc_audio  # noqa: F401
    import run_orchestrator  # noqa: F401
    import runtime_versions  # noqa: F401


def test_console_callables_exist() -> None:
    import main as main_mod
    import run_orchestrator as ro

    assert callable(ro.main)
    assert callable(main_mod.main)
