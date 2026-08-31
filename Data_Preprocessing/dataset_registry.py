"""Registry connecting dataset names to their native-layout adapters."""

try:
    from .datasets.bh import BhAdapter
    from .datasets.cohface import CohfaceAdapter
    from .datasets.pure import PureAdapter
    from .datasets.tokyotech import TokyoTechAdapter
    from .datasets.ubfc import UbfcAdapter
    from .datasets.ubfc_phys import UbfcPhysAdapter
except ImportError:
    from datasets.bh import BhAdapter
    from datasets.cohface import CohfaceAdapter
    from datasets.pure import PureAdapter
    from datasets.tokyotech import TokyoTechAdapter
    from datasets.ubfc import UbfcAdapter
    from datasets.ubfc_phys import UbfcPhysAdapter


DATASET_REGISTRY = {
    "PURE": PureAdapter(),
    "UBFC": UbfcAdapter(),
    "BH": BhAdapter(),
    "UBFC_PHYS": UbfcPhysAdapter(),
    "COHFACE": CohfaceAdapter(),
    "TOKYOTECH": TokyoTechAdapter(),
}


def get_adapter(dataset_name: str):
    name = dataset_name.upper()
    try:
        return DATASET_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {dataset_name!r}. Registered: {available}") from error
