from . import acoustic, cardrf, dronerf, drff_r2, noisy_rf, rfuav, rma, uavsig
from .tables import read, validate, write

BUILDERS = {
    "cardrf": cardrf.build, "dronerf": dronerf.build, "drff_r2": drff_r2.build, "noisy_rf": noisy_rf.build, "rfuav": rfuav.build,
    "rma": rma.build, "uavsig": uavsig.build,
    "ddl": acoustic.build_ddl, "dronenoise": acoustic.build_dronenoise, "esc50": acoustic.build_esc50,
    "miesikowska_uav": acoustic.build_miesikowska, "svanstrom": acoustic.build_svanstrom,
    "uavirbase": acoustic.build_uavirbase,
}

__all__ = ["BUILDERS", "read", "validate", "write"]
