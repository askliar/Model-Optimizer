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

"""Check that launcher utilities preserve the shared package source."""

import subprocess
from pathlib import Path


def test_repeated_launch_keeps_package_source_unchanged(tmp_path):
    """Source the launcher twice without changing the package version."""
    script = Path(__file__).resolve().parents[1] / "common/service_utils.sh"
    package = tmp_path / "modules/Model-Optimizer/modelopt/__init__.py"
    package.parent.mkdir(parents=True)
    original = b"__version__ = 'test-version'\n"
    package.write_bytes(original)
    subprocess.run(
        [
            "bash",
            "-ec",
            'function nvidia-smi { echo 4; }; source "$1"; source "$1"',
            "test",
            str(script),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert package.read_bytes() == original
