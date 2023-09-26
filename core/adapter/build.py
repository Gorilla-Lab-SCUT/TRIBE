from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .TRIBE import TRIBE
from .lame import LAME
from .tent import TENT
from .pl import PL
from .bn import BN
from .note import NOTE
from .ttac import TTAC
from .eata import EATA
from .cotta import CoTTA
from .petal import PETALFim


def build_adapter(cfg) -> type(BaseAdapter):
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    elif cfg.ADAPTER.NAME == "tribe":
        return TRIBE
    elif cfg.ADAPTER.NAME == "lame":
        return LAME
    elif cfg.ADAPTER.NAME == "tent":
        return TENT
    elif cfg.ADAPTER.NAME == "pl":
        return PL
    elif cfg.ADAPTER.NAME == "bn":
        return BN
    elif cfg.ADAPTER.NAME == "note":
        return NOTE
    elif cfg.ADAPTER.NAME == "ttac":
        return TTAC
    elif cfg.ADAPTER.NAME == "eata":
        return EATA
    elif cfg.ADAPTER.NAME == "cotta":
        return CoTTA
    elif cfg.ADAPTER.NAME == "petal":
        return PETALFim
    else:
        raise NotImplementedError("Implement your own adapter")

