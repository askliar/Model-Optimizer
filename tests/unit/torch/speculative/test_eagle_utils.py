# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regression tests for the speculative decoding Trainer loss contract."""

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import TrainingArguments
from transformers.utils import ModelOutput


@dataclass
class _DraftOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    train_acc: list | None = None
    eagle_loss: torch.Tensor | None = None
    preservation_loss: torch.Tensor | None = None


class _DraftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels, **kwargs):
        logits = input_ids.float() * self.weight
        loss = (logits - labels.float()).square().mean()
        return _DraftOutput(loss, logits, [1.0], loss.detach(), loss.detach() * 0)


@pytest.fixture
def trainer_cls(monkeypatch):
    example_path = Path(__file__).resolve().parents[4] / "examples/speculative_decoding"
    monkeypatch.syspath_prepend(str(example_path))
    # The example uses imports relative to its script directory.
    return importlib.import_module("eagle_utils").EagleTrainerWithAccLog


def _trainer(trainer_cls, tmp_path, accumulation_steps=1):
    return trainer_cls(
        model=_DraftModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=accumulation_steps,
        ),
    )


def test_evaluate_preserves_training_metrics(trainer_cls, tmp_path):
    trainer = _trainer(trainer_cls, tmp_path)
    batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[0, 0]])}
    trainer.model.train()
    trainer.compute_loss(trainer.model, batch.copy())
    training_accs = list(trainer.state.training_accs)
    component_losses = {key: list(value) for key, value in trainer.state.component_losses.items()}

    metrics = trainer.evaluate(
        [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([0, 0])}]
    )

    assert metrics["eval_loss"] == pytest.approx(2.5)
    assert trainer.state.training_accs == training_accs
    assert trainer.state.component_losses == component_losses


def test_training_step_normalizes_accumulated_gradients(trainer_cls, tmp_path):
    trainer = _trainer(trainer_cls, tmp_path, accumulation_steps=2)
    inputs = torch.tensor([[1, 2], [3, 4]])
    labels = torch.zeros_like(inputs)
    reference = _DraftModel()
    reference(input_ids=inputs, labels=labels).loss.backward()
    trainer.current_gradient_accumulation_steps = 2

    for index in range(2):
        with trainer.accelerator.accumulate(trainer.model):
            trainer.training_step(
                trainer.model,
                {"input_ids": inputs[index : index + 1], "labels": labels[index : index + 1]},
                num_items_in_batch=inputs.numel(),
            )

    torch.testing.assert_close(trainer.model.weight.grad, reference.weight.grad)
