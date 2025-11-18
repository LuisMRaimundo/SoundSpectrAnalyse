# log_config.py
import logging
from typing import Optional

def configure_root_logger(level: int = logging.INFO,
                          fmt: Optional[str] = None) -> None:
    """
    Configura o root logger exactamente uma vez.

    Parameters
    ----------
    level : int
        Nível mínimo de mensagens (default = INFO).
    fmt : str | None
        Formato da mensagem.  Se None, usa um formato padrão.
    """
    root = logging.getLogger()            # root logger
    if root.handlers:                     # já existe configuração → não duplica
        return

    if fmt is None:
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

    logging.basicConfig(level=level, format=fmt)

