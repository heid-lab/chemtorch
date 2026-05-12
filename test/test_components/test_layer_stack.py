from functools import partial

import torch
from omegaconf import OmegaConf

from chemtorch.components.layer.layer_stack import LayerStack


def test_layer_stack_accepts_hydra_config():
    cfg = OmegaConf.create(
        {
            "_target_": "torch.nn.Linear",
            "in_features": 2,
            "out_features": 2,
        }
    )

    stack = LayerStack(layer=cfg, depth=3)

    assert len(stack.layers) == 3
    assert all(isinstance(layer, torch.nn.Linear) for layer in stack.layers)
    assert len({id(layer) for layer in stack.layers}) == 3


def test_layer_stack_accepts_partial_factory():
    stack = LayerStack(layer=partial(torch.nn.Linear, 2, 2), depth=3)

    assert len(stack.layers) == 3
    assert all(isinstance(layer, torch.nn.Linear) for layer in stack.layers)
    assert len({id(layer) for layer in stack.layers}) == 3


def test_layer_stack_can_share_factory_layer():
    stack = LayerStack(
        layer=partial(torch.nn.Linear, 2, 2),
        depth=3,
        share_weights=True,
    )

    assert len(stack.layers) == 3
    assert len({id(layer) for layer in stack.layers}) == 1


def test_layer_stack_deepcopies_module_instances_by_default():
    layer = torch.nn.Linear(2, 2)

    stack = LayerStack(layer=layer, depth=3)

    assert len(stack.layers) == 3
    assert all(isinstance(layer, torch.nn.Linear) for layer in stack.layers)
    assert len({id(layer) for layer in stack.layers}) == 3
    assert all(stacked_layer is not layer for stacked_layer in stack.layers)


def test_layer_stack_can_share_module_instance():
    layer = torch.nn.Linear(2, 2)

    stack = LayerStack(layer=layer, depth=3, share_weights=True)

    assert len(stack.layers) == 3
    assert len({id(stacked_layer) for stacked_layer in stack.layers}) == 1
    assert stack.layers[0] is layer

