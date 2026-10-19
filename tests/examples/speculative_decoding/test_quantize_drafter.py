# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for drafter quantization configuration."""

import importlib.util
from pathlib import Path

import pytest
import torch.nn as nn

import modelopt.torch.quantization as mtq

_SPEC = importlib.util.spec_from_file_location(
    "quantize_drafter",
    Path(__file__).parents[3] / "examples/speculative_decoding/scripts/quantize_drafter.py",
)
assert _SPEC is not None and _SPEC.loader is not None
quantize_drafter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quantize_drafter)


@pytest.mark.parametrize("qformat", ["w4a16_nvfp4", "nvfp4", "fp8"])
@pytest.mark.parametrize(
    ("quantize_lm_head", "exclude", "enabled"),
    [(False, [], False), (True, [], True), (True, ["*lm_head*"], False)],
)
def test_lm_head_quantizer_configuration(qformat, quantize_lm_head, exclude, enabled):
    """LM-head quantizers follow the selected format and explicit exclusions."""
    model = nn.ModuleDict({"lm_head": nn.Linear(16, 16, bias=False)})
    quant_cfg = quantize_drafter.build_quant_cfg(qformat, exclude, quantize_lm_head)
    mtq.replace_quant_module(model)
    mtq.set_quantizer_by_cfg(model, quant_cfg["quant_cfg"])

    assert model["lm_head"].weight_quantizer.is_enabled == enabled
    assert model["lm_head"].input_quantizer.is_enabled == (enabled and qformat != "w4a16_nvfp4")
