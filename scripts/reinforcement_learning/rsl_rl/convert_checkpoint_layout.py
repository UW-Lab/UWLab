# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a published OmniReset state expert from beta to EA observation order on CPU."""

from __future__ import annotations

import argparse
import hashlib
import torch
from itertools import chain
from pathlib import Path
from tensordict import TensorDict

from rsl_rl.models import MLPModel

from uwlab_rl.rsl_rl.checkpoint_layout import convert_omnireset_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    raw = args.checkpoint.read_bytes()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    obs = TensorDict({"policy": torch.zeros(1, 43)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs, groups, "actor", 7, hidden_dims=[512, 256, 128, 64], activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution", "init_std": 1.0,
                          "std_type": "log", "std_range": [0.001, 2.0]},
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[512, 256, 128, 64], activation="elu", obs_normalization=True)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    critic.load_state_dict(saved["critic_state_dict"], strict=True)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()))
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    converted = convert_omnireset_checkpoint(saved, actor, critic, optimizer, hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(converted, stream)
    print(f"Saved EA-layout checkpoint to {args.output}; original unchanged: {args.checkpoint}")


if __name__ == "__main__":
    main()
