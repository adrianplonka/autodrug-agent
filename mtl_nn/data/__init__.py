# Featurization only needs RDKit/NumPy and is required for inference, so it
# is always imported. The data-loading stack (tdc_loader, dataset) pulls in
# heavy training-only deps (PyTDC, scikit-learn); import it lazily so that
# pure inference works on machines that don't have the training extras.
from .featurizer import smiles_to_morgan_fp, batch_smiles_to_fps

__all__ = [
    "smiles_to_morgan_fp",
    "batch_smiles_to_fps",
]

try:
    from .tdc_loader import download_and_merge_tdc
    from .dataset import ADMETMultiTaskDataset, build_dataloaders
except ImportError as _exc:  # training extras (tdc / sklearn) not installed
    import warnings
    warnings.warn(
        f"mtl_nn.data: training utilities unavailable ({_exc}). "
        "Featurization still works; install 'tdc' and 'scikit-learn' for "
        "data download/training.",
        stacklevel=2,
    )
else:
    __all__ += [
        "download_and_merge_tdc",
        "ADMETMultiTaskDataset",
        "build_dataloaders",
    ]
