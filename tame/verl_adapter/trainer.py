"""TAME's trainer for verl: behavioral feedback, dynamic sampling and the SAE-penalized actor, on top of RayPPOTrainer.

TameTrainer.fit is verl v0.7.1's RayPPOTrainer.fit with three changes, each marked "TAME:":
  1. the generation block is replaced by _rollout_batch plus, per method, _dynamic_sampling (DAPO) or _tame_augment (TAME);
  2. the batch is trimmed to a multiple of the data-parallel size (the feedback loop makes the batch size vary);
  3. the advantages of second-attempt and distilled rows are corrected after verl's group advantage.

Behavioral feedback (method=tame, unless no_feedback), per step:
  * every update_every steps the prompt updater U revises the refinement prompt p_e from recent attempts and rewards;
  * first attempts with reward < tau_r are regenerated under p_e, conditioned on the first attempt and its reward;
  * second attempts that beat their first attempt are re-paired with the ORIGINAL prompt as "distilled" rows whose
    advantage is the reward, a reward-weighted cross-entropy in the style of RAFT;
  * GRPO advantages are computed per (prompt, first) and (prompt, second) group through their uid.
"""

import copy
import os
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from ..feedback import PromptEvolver
from .config import from_env


def resplice_position_ids(position_ids, prompt_len, response_len):
    """Position ids for [prompt | new response]: keep the prompt part, continue counting after its last token.

    Works for (batch, seq) text positions and (batch, channels, seq) mrope positions; response tokens are text, so every
    channel advances by one per token after the last prompt token.
    """
    last = position_ids[..., prompt_len - 1:prompt_len]
    ramp = torch.arange(1, response_len + 1, device=position_ids.device, dtype=position_ids.dtype)
    return torch.cat([position_ids[..., :prompt_len], last + ramp], dim=-1)


def splice_distilled(original, second):
    """Rows pairing the ORIGINAL prompt with the response of a better second attempt (the RAFT consolidation set)."""
    out = original.select_idxs(list(range(len(original))))
    prompt_len = original.batch["prompts"].shape[1]
    response_len = second.batch["responses"].shape[1]
    out.batch["responses"] = second.batch["responses"].clone()
    out.batch["input_ids"] = torch.cat([original.batch["prompts"], second.batch["responses"]], dim=1)
    out.batch["attention_mask"] = torch.cat(
        [original.batch["attention_mask"][:, :prompt_len], second.batch["attention_mask"][:, prompt_len:]], dim=1)
    out.batch["position_ids"] = resplice_position_ids(original.batch["position_ids"], prompt_len, response_len)
    out.batch["rm_scores"] = second.batch["rm_scores"].clone()
    out.batch["response_mask"] = compute_response_mask(out)
    return out


def informative_indices(uids, rewards):
    """Row indices of the groups whose rewards are not all equal, keeping the groups contiguous and in order."""
    groups = {}
    for i, uid in enumerate(uids):
        groups.setdefault(uid, []).append(i)
    return [i for g in groups.values() if len({round(float(rewards[j]), 6) for j in g}) > 1 for i in g]


def fix_advantages(kind, uids, rewards, advantages, response_mask):
    """verl's GRPO advantage gives a singleton group the advantage score / 1. tame.train gives distilled rows
    max(reward, 0) and a singleton second-attempt group 0, so reproduce that."""
    adv = advantages.clone()
    mask = response_mask.to(adv.dtype)
    second_size = {}
    for k, u in zip(kind, uids):
        if k == "second":
            second_size[u] = second_size.get(u, 0) + 1
    for i, (k, u) in enumerate(zip(kind, uids)):
        if k == "distilled":
            adv[i] = max(float(rewards[i]), 0.0) * mask[i]
        elif k == "second" and second_size[u] == 1:
            adv[i] = 0.0
    return adv


class TameTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tame_cfg = from_env()
        assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, "TameTrainer does not support REMAX"
        assert self.config.actor_rollout_ref.rollout.n > 1, "GRPO needs rollout.n > 1"
        if self.tame_cfg["gamma"] > 0:
            assert self.ref_in_actor, "TAME's SAE penalty compares with the pre-RL model: enable LoRA (model.lora_rank > 0)"
        self.evolver = None if self.tame_cfg["no_feedback"] else PromptEvolver(
            self.tame_cfg["dataset"], self.tame_cfg["updater"])
        self._extra_iter = None

    # ------------------------------------------------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------------------------------------------------
    def _rollout_batch(self, batch_dict, timing_raw, curr_step_profile):
        """verl's per-step generation: returns the batch with responses and rewards, and the repeated generation inputs."""
        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        n = self.config.actor_rollout_ref.rollout.n
        gen_input = gen_batch.repeat(repeat_times=n, interleave=True)

        with marked_timer("gen", timing_raw, color="red"):
            if curr_step_profile:
                self.async_rollout_manager.start_profile()
            gen_output = self.async_rollout_manager.generate_sequences(gen_input)
            self.checkpoint_manager.sleep_replicas()
            if curr_step_profile:
                self.async_rollout_manager.stop_profile()
            timing_raw.update(gen_output.meta_info["timing"])
            gen_output.meta_info.pop("timing", None)

        batch = batch.repeat(repeat_times=n, interleave=True)
        batch = batch.union(gen_output)
        batch.batch["response_mask"] = compute_response_mask(batch)
        return batch, gen_input

    def _decode_responses(self, batch):
        responses, mask = batch.batch["responses"], batch.batch["response_mask"].bool()
        return [self.tokenizer.decode(responses[i][mask[i]], skip_special_tokens=True) for i in range(len(batch))]

    # ------------------------------------------------------------------------------------------------------------
    # DAPO: dynamic sampling
    # ------------------------------------------------------------------------------------------------------------
    def _next_extra_batch(self):
        if self._extra_iter is None:
            self._extra_iter = iter(self.train_dataloader)
        try:
            return next(self._extra_iter)
        except StopIteration:
            self._extra_iter = iter(self.train_dataloader)
            return next(self._extra_iter)

    def _informative(self, batch):
        idx = informative_indices(batch.non_tensor_batch["uid"], batch.batch["rm_scores"].sum(-1).tolist())
        return batch.select_idxs(idx) if idx else None

    def _dynamic_sampling(self, batch, timing_raw):
        """Keep prompts whose rollout group has reward variance; draw extra prompts until the batch is full."""
        n = self.config.actor_rollout_ref.rollout.n
        rows_needed = self.config.data.train_batch_size * n
        parts = [p for p in [self._informative(batch)] if p is not None]
        for _ in range(self.tame_cfg["dynamic_sampling_rounds"]):
            if sum(len(p) for p in parts) >= rows_needed:
                break
            extra, _ = self._rollout_batch(self._next_extra_batch(), timing_raw, False)
            part = self._informative(extra)
            if part is not None:
                parts.append(part)
        if not parts:
            return batch   # no group had reward variance at all: keep the plain batch instead of an empty one
        kept = DataProto.concat(parts) if len(parts) > 1 else parts[0]
        return kept.select_idxs(list(range(min(len(kept), rows_needed))))

    # ------------------------------------------------------------------------------------------------------------
    # TAME: behavioral feedback
    # ------------------------------------------------------------------------------------------------------------
    def _tame_augment(self, batch, gen_input, timing_raw, curr_step_profile, metrics):
        cfg = self.tame_cfg
        n = len(batch)
        rewards = batch.batch["rm_scores"].sum(-1)
        texts = self._decode_responses(batch)
        questions = [batch.non_tensor_batch["extra_info"][i]["question"] for i in range(n)]
        batch.non_tensor_batch["tame_kind"] = np.array(["first"] * n, dtype=object)

        self.evolver.history += list(zip(questions, texts, rewards.tolist()))
        if self.global_steps % cfg["update_every"] == 0:
            self.evolver.update()

        gated = [i for i in range(n) if rewards[i].item() < cfg["tau_r"]]
        metrics.update({"tame/first_reward": rewards.mean().item(), "tame/gated_frac": len(gated) / max(n, 1)})
        if not gated:
            return batch

        # second attempts: same prompts, but the system prompt is p_e plus the first attempt and its reward
        gen2 = gen_input.select_idxs(gated)
        prompts = np.empty(len(gated), dtype=object)
        for slot, i in enumerate(gated):
            messages = copy.deepcopy(list(gen2.non_tensor_batch["raw_prompt"][slot]))
            system = self.evolver.system_for(questions[i], texts[i], rewards[i].item())
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = system
            else:
                messages.insert(0, {"role": "system", "content": system})
            prompts[slot] = messages
        gen2.non_tensor_batch["raw_prompt"] = prompts
        with marked_timer("gen_second", timing_raw, color="red"):
            out2 = self.async_rollout_manager.generate_sequences(gen2)
            self.checkpoint_manager.sleep_replicas()
            out2.meta_info.pop("timing", None)

        second = batch.select_idxs(gated)
        uids = list(second.non_tensor_batch["uid"])
        second.pop(batch_keys=[k for k in out2.batch.keys() if k in second.batch.keys()],
                   non_tensor_batch_keys=[k for k in out2.non_tensor_batch.keys() if k in second.non_tensor_batch])
        second = second.union(out2)
        second.batch["response_mask"] = compute_response_mask(second)
        second.non_tensor_batch["uid"] = np.array([f"{u}#second" for u in uids], dtype=object)
        second.non_tensor_batch["tame_kind"] = np.array(["second"] * len(gated), dtype=object)

        # distilled: refinements that beat the first attempt, re-paired with the original prompt
        rewards2 = second.batch["rm_scores"].sum(-1)
        better = [k for k, i in enumerate(gated) if rewards2[k].item() > rewards[i].item()]
        parts = [batch, second]
        if better:
            distilled = splice_distilled(batch.select_idxs([gated[k] for k in better]), second.select_idxs(better))
            distilled.non_tensor_batch["uid"] = np.array(
                [f"{u}#distilled#{j}" for j, u in enumerate(distilled.non_tensor_batch["uid"])], dtype=object)
            distilled.non_tensor_batch["tame_kind"] = np.array(["distilled"] * len(better), dtype=object)
            parts.append(distilled)
        metrics.update({"tame/second_reward": rewards2.mean().item(), "tame/second_n": len(gated),
                        "tame/refined_kept": len(better) / max(n, 1)})

        # align keys, drop the rollout log-probs that do not describe the spliced rows, and merge
        common_t = set.intersection(*[set(p.batch.keys()) for p in parts]) - {"rollout_log_probs", "routed_experts"}
        common_n = set.intersection(*[set(p.non_tensor_batch.keys()) for p in parts])
        meta = dict(batch.meta_info)
        aligned = []
        for p in parts:
            p = p.select(batch_keys=sorted(common_t), non_tensor_batch_keys=sorted(common_n))
            p.meta_info = dict(meta)
            aligned.append(p)
        return DataProto.concat(aligned)

    def _trim_to_dp_multiple(self, batch):
        """The workers split the batch evenly across data-parallel ranks. Drop trailing rows (distilled first)."""
        dp = self._get_dp_size(self.actor_rollout_wg, "actor")
        keep = len(batch) - len(batch) % dp
        return batch if keep == len(batch) else batch.select_idxs(list(range(keep)))

    def _tame_fix_advantages(self, batch):
        if "tame_kind" not in batch.non_tensor_batch:
            return batch
        rewards = batch.batch["token_level_rewards"].sum(-1).tolist()
        adv = fix_advantages(batch.non_tensor_batch["tame_kind"], batch.non_tensor_batch["uid"], rewards,
                             batch.batch["advantages"], batch.batch["response_mask"])
        batch.batch["advantages"] = adv
        batch.batch["returns"] = adv
        return batch

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if self.evolver is not None:
            path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "refinement_prompt.txt"), "w") as f:
                f.write(self.evolver.prompt)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._extra_iter = None  # TAME: second iterator over the train set for dynamic sampling

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # TAME: rollout, dynamic sampling (DAPO) and behavioral feedback (TAME) replace verl's generation block
                    batch, gen_input = self._rollout_batch(batch_dict, timing_raw, curr_step_profile)
                    if self.tame_cfg["method"] == "dapo":
                        batch = self._dynamic_sampling(batch, timing_raw)
                    elif self.evolver is not None:
                        batch = self._tame_augment(batch, gen_input, timing_raw, curr_step_profile, metrics)
                    batch = self._trim_to_dp_multiple(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        batch = self._tame_fix_advantages(batch)  # TAME: advantages of second and distilled rows

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
