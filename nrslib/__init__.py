from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import dotenv
from .src import utils
from .src.training_pipeline import train
from .src.testing_pipeline import test
from os.path import abspath, expanduser, dirname, join
import shutil
import hydra
from pytorch_lightning import (
    LightningModule
)

dotenv.load_dotenv(override=True)


def start_train(config_dir, overrides=[], config_name="train", job_name="train"):
    """Start Training
    Args:
        config_dir (str): Configuration directory
        overrides (list): List containing override configurations
        config_name (str): 'train' in case of training run
        job_name (str): 'train' in case of training run

    Returns None
    """
    try:
        initialize_config_dir(
            config_dir=abspath(expanduser(config_dir)),
            version_base=None,
            job_name=job_name,
        )
        config = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )

        # Applies optional utilities
        utils.extras(config)

        # Train model
        return train(config)
    finally:
        GlobalHydra.instance().clear()


def start_test(config_dir, overrides=[], config_name="test", job_name="test"):
    """Start Testing
    Args:
        config_dir (str): Configuration directory
        overrides (list): List containing override configurations
        config_name (str): 'test' in case of test run
        job_name (str): 'test' in case of test run

    Returns None
    """
    try:
        initialize_config_dir(
            config_dir=abspath(expanduser(config_dir)),
            version_base=None,
            job_name=job_name,
        )
        config = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )

        # Applies optional utilities
        utils.extras(config)

        # Evaluate model
        return test(config)
    finally:
        GlobalHydra.instance().clear()


def export_default_config(output_dir):
    """Exports the default configuration. Useful when small modifications to the default configuartions are required.

    Args:
        output_dir (str): Path to save configuration files in

    Returns None
    """
    f = dirname(abspath(__file__))
    input = join(f, "configs")
    output = abspath(expanduser(output_dir))
    if output != input:
        shutil.copytree(input, output)

def start_test_baseline(config_dir, overrides=[],config_name="BERT"):
    """Baseline evaluation using BERT
    Args:
        config_dir (str): Configuration directory
        overrides (list): List containing override configurations
        config_name (str): 'BERT' in case of Baseline run

    Returns None
    """
    try:
        initialize_config_dir(
            config_dir=abspath(expanduser(config_dir)),
            version_base=None
        )
        config = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )
        model: LightningModule = hydra.utils.instantiate(
            config.model, _recursive_=False, _convert_="partial"
        )

        # Evaluate model
        return model.run_pipline()
    finally:
        GlobalHydra.instance().clear()
