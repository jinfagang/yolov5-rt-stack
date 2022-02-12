# Copyright (c) 2021, yolort team. All rights reserved.

from typing import Dict, Callable, Optional, Tuple

import torch
from torch import nn, Tensor


@torch.no_grad()
def get_trace_module(model_func: Callable[..., nn.Module], input_sample: Optional[Tensor] = None):
    """
    Get the tracing of a given model function.

    Example:

        >>> from yolort.models import yolov5s
        >>> from yolort.relaying.trace_wrapper import get_trace_module
        >>>
        >>> model = yolov5s(pretrained=True)
        >>> tracing_module = get_trace_module(model)
        >>> print(tracing_module.code)
        def forward(self,
            x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
          _0, _1, _2, = (self.model).forward(x, )
          return (_0, _1, _2)

    Args:
        model_func (Callable): The model function to be traced.
        input_sample (Tensor, optional): An input for tracing. Default: None.
    """

    if input_sample is None:
        input_sample = torch.rand(1, 3, 640, 640)
    input_sample = input_sample.to(device=model_func.device)

    model = TraceWrapper(model_func)
    model.eval()

    trace_module = torch.jit.trace(model, input_sample)
    trace_module.eval()

    return trace_module


def dict_to_tuple(out_dict: Dict[str, Tensor]) -> Tuple:
    """
    Convert the model output dictionary to tuple format.
    """
    if "masks" in out_dict.keys():
        return (
            out_dict["boxes"],
            out_dict["scores"],
            out_dict["labels"],
            out_dict["masks"],
        )
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(nn.Module):
    """
    This is a wrapper for `torch.jit.trace`, as there are some scenarios
    where `torch.jit.script` support is limited.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return dict_to_tuple(out[0])
