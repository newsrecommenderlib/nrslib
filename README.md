# nrslib

This library is a modular, end-to-end pipeline for news recommender systems.

## Quick Start

### Installation

Install with pip

```shell
pip install nrslib
```

### Main Technologies

Familarity with the follow tools are recommended to use this library.

[PyTorch Lightning](https://www.pytorchlightning.ai/) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://hydra.cc/) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

[TorchMetrics](https://torchmetrics.readthedocs.io/) - a collection of 80+ PyTorch metrics implementations and an easy-to-use API to create custom metrics.

[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) - A clean and scalable template to kickstart your deep learning project.

### Running a model

Import the library

```python
import nrslib
```

Export default configs

```python
nrslib.export_default_config(“~/configs”)
```

Choose and edit configs in the open console dialog and start training afterwards

```python
nrslib.start_train(“~/configs”, [“experiment=naml.yaml”])
```

To test a model, run

```python
nrslib.start_test(“~/configs”, [“experiment=naml.yaml”])
```

To extend a model or datamodule inherit it

```python
from nrslib.src.models.naml import NAML
class ImprovedNAML(NAML):
```

Then adjust the configurations to use the new class

```yaml
_target_: path.to.class.ImprovedNAML
```

### Algrothims

There are multuiple algrothims implemented in the library and ready to work with the MIND dataset. The algrothims supported are: 
- Baselines Models  
    - Neural Collaborative Filtering
    - Content Based (BERT)
- Advanced Models
    - MKR
    - RippleNet
    - DKN
    - LSTUR
    - NAML
    - NRMS