"""Train TAME, GRPO or DAPO with verl.

python -m tame.verl_adapter.main <verl overrides> +tame.method=tame +tame.dataset=virl \
    +tame.sae=sae/virl +tame.features=analysis/virl_step200/features.json

Everything after the module name is a verl Hydra override. The TAME settings are the +tame.* keys of tame.verl_adapter.config.
scripts/train_verl.sh sets all of them.
"""

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device

from . import config as C

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS_THROUGH = ("OPENROUTER_API_KEY", "LLM_API_KEY", "LLM_BASE_URL")


class TameTaskRunner(TaskRunner):
    """verl's TaskRunner with TAME's worker and trainer."""

    def add_actor_rollout_worker(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        from .worker import TameActorRolloutRefWorker

        _, worker_group_cls = super().add_actor_rollout_worker(config)
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(TameActorRolloutRefWorker)
        return TameActorRolloutRefWorker, worker_group_cls

    def run(self, config):
        import verl.trainer.main_ppo as main_ppo

        from .trainer import TameTrainer

        main_ppo.RayPPOTrainer = TameTrainer
        super().run(config)


def check_verl_config(config):
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("TAME on verl needs actor_rollout_ref.actor.strategy=fsdp or fsdp2")
    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError("TAME on verl needs the FSDP workers: set trainer.use_legacy_worker_impl=enable")


def ray_env_vars(tame_cfg, existing=None, environ=None):
    """Environment shipped to every Ray process: the TAME settings, the import path and the LLM credentials."""
    environ = os.environ if environ is None else environ
    env = dict(existing or {})
    env[C.ENV_VAR] = C.dumps(tame_cfg)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (REPO_ROOT, environ.get("PYTHONPATH", "")) if p)
    env.update({name: environ[name] for name in PASS_THROUGH if name in environ})
    return env


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    check_verl_config(config)
    tame_cfg = C.resolve(OmegaConf.to_container(config.get("tame") or {}, resolve=True))
    existing = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.env_vars") or {}
    OmegaConf.set_struct(config, False)
    OmegaConf.update(config, "ray_kwargs.ray_init.runtime_env.env_vars",
                     ray_env_vars(tame_cfg, OmegaConf.to_container(existing)), force_add=True)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TameTaskRunner))


if __name__ == "__main__":
    main()
