import logging
import os
from collections import defaultdict
from typing import Any, cast, Dict, IO, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import HTTPURLHandler, PathManager
from termcolor import colored
from torch.nn.parallel import DataParallel, DistributedDataParallel


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

__all__ = ["Checkpointer", "PeriodicCheckpointer"]


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


def _load_model(net, checkpoint_state_dict) -> _IncompatibleKeys:
    """
    Load weights from a checkpoint.

    Args:
        checkpoint (Any): checkpoint contains the weights.

    Returns:
        ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
            and ``incorrect_shapes`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
            * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

        This is just like the return value of
        :func:`torch.nn.Module.load_state_dict`, but with extra support
        for ``incorrect_shapes``.
    """
    #checkpoint_state_dict = checkpoint.pop("model")
    #self._convert_ndarray_to_tensor(checkpoint_state_dict)

    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching.
    #_strip_prefix_if_present(checkpoint_state_dict, "module.")

    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = net.state_dict()

    #checkpoint_state_dict = checkpoint #.pop("model")

    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(
                model_param, nn.parameter.UninitializedParameter
            ):
                continue
            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:

                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(
                        model: torch.nn.Module, key: str
                    ) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(net, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = net.load_state_dict(checkpoint_state_dict, strict=False)
    print("!!!! incorrect shapes ", incorrect_shapes)
    return len(incorrect_shapes)

    return _IncompatibleKeys(
        missing_keys=incompatible.missing_keys,
        unexpected_keys=incompatible.unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )


def update_network(net, prefix="criterion"):
    from collections import OrderedDict
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            continue
        
        net_[k] = net[k]
    return net_


def load_with_key_mismatch(net, checkpoint_state_dict, update_name= "net_Latent"):
    model_state_dict = net.state_dict()
    model_key = model_state_dict.keys()
    check_key = checkpoint_state_dict.keys()

    from collections import OrderedDict    
    updated_checkpoint = OrderedDict()

    for mk in model_key:
        if mk in check_key:
            updated_checkpoint[mk] = checkpoint_state_dict[mk] 
            continue
        
        if mk.split(".")[0].startswith("netD"): continue
        str_ck = mk.replace(mk.split(".")[0] + ".", "")
        updated_checkpoint[mk] = checkpoint_state_dict[str_ck]

    _load_model(net, updated_checkpoint)
    #net.load_state_dict(updated_checkpoint, strict=False)
    return
