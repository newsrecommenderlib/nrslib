Quick start
=====================================================

Main Technologies
------------------
Familarity with the follow tools are recommended to use this library.

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

`Hydra <https://hydra.cc/>`_ - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

`TorchMetrics <https://torchmetrics.readthedocs.io/>`_ - a collection of 80+ PyTorch metrics implementations and an easy-to-use API to create custom metrics.

`Lightning-Hydra-Template <https://github.com/ashleve/lightning-hydra-template>`_ - A clean and scalable template to kickstart your deep learning project.

Running a model
------------------
Import the library::

    import nrslib

Export default configs::

    nrslib.export_default_config(“~/configs”)

Choose and edit configs in the open console dialog and start training afterwards::

    nrslib.start_train(“~/configs”, [“experiment=naml.yaml”])

To test a model, run::

    nrslib.start_test(“~/configs”, [“experiment=naml.yaml”])


To extend a model or datamodule inherit it::

    from nrslib.src.models.naml import NAML
    class ImprovedNAML(NAML):
        . . . .

Then adjust the configurations to use the new class::

    _target_: path.to.class.ImprovedNAML



Using Torchmetrics
------------------
Metrics that should be used for model evaluation and testing need to be added to the models .yaml configuration file.
Example: The .yaml file for the Neural Collaborative Filtering model could look as follows::

    _target_: src.models.collaborativeFiltering.CollaborativeFiltering
    lr: 0.001
    n_factors: 100
    data_dir: ${data_dir}/MIND
    metrics:
      valid:
        accuracy:
          _target_: torchmetrics.Accuracy
        ndcg5:
          _target_: torchmetrics.RetrievalNormalizedDCG
          k: 5
      test:
        accuracy:
          _target_: torchmetrics.Accuracy
        ndcg5:
          _target_: torchmetrics.RetrievalNormalizedDCG
          k: 5

