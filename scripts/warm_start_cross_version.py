#!/usr/bin/env python3
"""Cross-version warm start: copy a source checkpoint's shared weights into a
fresh agent built from a TARGET config (different ppo_network_type), via a
strict=False load. New layers keep their (zero-)init.

Use case: seed a v9 run from run 004's v8 final checkpoint — v9's
``action_econ_proj`` is zero-init, so the warm-started agent behaves identically
to 004 at step 0 and only has to learn the economy pathway, instead of training
10M fresh steps.

The output is a normal checkpoint stamped with the TARGET network type, so the
distributed trainer's loader rebuilds the right (target) net on resume:

    python scripts/warm_start_cross_version.py \
        --config configs/bglike/ppo_v9_econ_74257.yaml \
        --source runs/bglike/ppo_new_004/checkpoints/dist_bglike_ppo_v8_dist_final.pt \
        --out runs/bglike/ppo_new_006/checkpoints/warm_from_004.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import src.envs  # noqa: F401
from src.config.schema import load_config
from src.registry import make_agent
from src.training.obs_sizing import apply_bg_observation_defaults
from src.training.patch_config import apply_patch_to_agent_params


def _build_target_agent(config_path: Path, device: str):
    """Mirror run_distributed's sizing so the agent kwargs match a real run."""
    app = load_config(config_path)
    game_id = str(app.game.id)
    game_params = dict(app.game.params or {})
    game_params["game_id"] = game_id
    agent_params = dict(app.agent.params)
    agent_params["device"] = device

    if game_id == "bglike":
        from src.envs.bglike.action_map import NUM_ENV_ACTIONS

        game_params.setdefault("obs_kind", "bglike_v5")
        apply_patch_to_agent_params(game_params, agent_params)
        apply_bg_observation_defaults(
            game_id, agent_params, obs_kind=game_params.get("obs_kind")
        )
        num_actions = int(NUM_ENV_ACTIONS)
    else:
        raise SystemExit(f"only bglike supported, got game id {game_id!r}")

    agent_params["num_actions"] = num_actions
    agent = make_agent(app.agent.id, **agent_params)
    return agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="TARGET config (e.g. v9)")
    ap.add_argument("--source", type=Path, required=True, help="SOURCE checkpoint (e.g. v8)")
    ap.add_argument("--out", type=Path, required=True, help="output warm-started checkpoint")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    agent = _build_target_agent(args.config, args.device)
    target_net = agent.policy_net
    tgt_type = agent._ppo_network_type

    raw = torch.load(str(args.source), map_location=args.device)
    src_type = raw.get("ppo_network_type", "?")
    src_sd = raw.get("policy_state_dict")
    if src_sd is None:
        raise SystemExit("source checkpoint has no policy_state_dict")

    missing, unexpected = target_net.load_state_dict(src_sd, strict=False)
    # Shapes must line up on every shared key; the only acceptable "missing"
    # keys are genuinely new target layers, and there must be no unexpected
    # source keys (that would mean the architectures diverged unexpectedly).
    if unexpected:
        raise SystemExit(
            f"source has {len(unexpected)} keys not in target net (arch mismatch): "
            f"{unexpected[:6]}"
        )
    print(f"warm start: {src_type} -> {tgt_type}")
    print(f"  shared keys loaded; new target-only keys kept at init: {missing or '(none)'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(args.out))
    saved = torch.load(str(args.out), map_location="cpu")
    print(f"  wrote {args.out}  (ppo_network_type={saved.get('ppo_network_type')})")


if __name__ == "__main__":
    main()
