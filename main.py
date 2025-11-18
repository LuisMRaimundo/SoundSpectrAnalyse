# main.py - Corrected and Consolidated Launcher

import sys
import traceback
import logging

# Attempt to import PyQt5, with a fallback to PySide6.
# This makes the application more flexible.
try:
    from PyQt5.QtWidgets import QApplication
except ModuleNotFoundError:
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        # If neither is found, exit with a helpful message.
        raise SystemExit(
            "Neither PyQt5 nor PySide6 were found. "
            "Please install one with: 'pip install PyQt5'"
        )

# Local imports
from log_config import configure_root_logger
from interface import SpectrumAnalyzer


def global_exception_hook(exc_type, exc_value, exc_traceback):
    """
    A global exception hook to catch any unhandled exceptions in the application.
    It logs the error and safely attempts to quit the application.
    """
    # Log the full traceback for debugging purposes
    logging.critical("An unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))

    # Also print to stderr for immediate visibility if logs aren't monitored
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    # Safely quit the application instance if it exists
    app = QApplication.instance()
    if app:
        app.quit()


def main():
    """
    Main entry point for the spectral analysis application.
    """
    # 1. Configure logging as the first step.
    configure_root_logger()

    # 2. Set the global exception hook for robust error handling.
    sys.excepthook = global_exception_hook

    # 3. Create and run the QApplication.
    app = QApplication(sys.argv)
    window = SpectrumAnalyzer()
    window.show()

    # 4. Start the application's event loop and exit when it's done.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()