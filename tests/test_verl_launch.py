import os
import shlex
import subprocess

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from tame.verl_adapter import config as C
from tame.verl_adapter.main import check_verl_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def overrides(method, dataset):
    env = dict(os.environ, DRY_RUN="1", METHOD=method, DATASET=dataset)
    out = subprocess.run(["bash", "scripts/train_verl.sh"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    argv = shlex.split(out.stdout)
    assert argv[:3] == ["python", "-m", "tame.verl_adapter.main"]
    return argv[3:]


def test_script_overrides_compose_with_verl_config():
    for method, dataset in [("tame", "virl"), ("tame", "spavl"), ("grpo", "virl"), ("dapo", "spavl")]:
        with initialize_config_module(config_module="verl.trainer.config", version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=overrides(method, dataset))
        check_verl_config(cfg)
        assert cfg.actor_rollout_ref.rollout.n == 8
        assert cfg.actor_rollout_ref.model.lora_rank == 64
        tame = C.resolve(OmegaConf.to_container(cfg.tame, resolve=True))
        assert tame["method"] == method and tame["dataset"] == dataset
        assert (tame["gamma"] > 0) == (method == "tame")


def test_extra_overrides_pass_through():
    env = dict(os.environ, DRY_RUN="1")
    out = subprocess.run(["bash", "scripts/train_verl.sh", "trainer.total_training_steps=3"], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.split()[-1] == "trainer.total_training_steps=3"
