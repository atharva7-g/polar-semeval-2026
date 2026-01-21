from pathlib import Path
from semevalpolar.utils import get_project_root
import os

BATCH_SIZE: int = 10

data_path: str = os.path.join(get_project_root(), 'data', 'training', 'eng.csv')

# ----------------------------------------------------------------------
# Example of how to expose a generator factory (optional)
# ----------------------------------------------------------------------
def get_training_generator(randomize: bool = True):
    """
    Convenience wrapper that creates the data generator using the
    configured path and batch size.

    Parameters
    ----------
    randomize : bool, optional
        Whether to shuffle the data each epoch.

    Returns
    -------
    generator
        The generator returned by `create_gen`.
    """
    from semevalpolar.llm.main import create_gen
    return create_gen(str(data_path), batch_size=BATCH_SIZE, randomize=randomize)
