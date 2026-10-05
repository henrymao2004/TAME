"""FSDP actor with TAME's representation-level objective.

TameActor extends verl's DataParallelPPOActor with the asymmetric SAE penalty
    L_SAE = sum_i ReLU(a_i^student - a_i^base)^2
over the template-associated latents of a frozen TopK SAE. The baseline activations a^base come from the pre-RL model,
which with LoRA is the actor itself with the adapter disabled (the same trick verl uses for the reference policy).
update_policy is a copy of verl v0.7.1 with the lines marked "TAME:" added.
"""

import contextlib
import json
import math

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor, logger

from ..sae import load_saes
from .config import from_env

_WRAPPER_TOKENS = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")


def find_module(model, key):
    """Resolve 'layers.12.self_attn.q_proj' inside the language model, through PEFT and FSDP wrappers."""
    suffix = "language_model." + key
    for name, module in model.named_modules():
        for token in _WRAPPER_TOKENS:
            name = name.replace(token, "")
        if name.endswith(suffix):
            return module
    raise KeyError(key)


class ActivationRecorder:
    """Keeps the autograd-connected output of the hooked modules while active."""

    def __init__(self, model, keys):
        self.out, self.active = {}, False
        self.handles = [find_module(model, k).register_forward_hook(self._hook(k)) for k in keys]

    def _hook(self, key):
        def fn(module, inputs, output):
            if self.active:
                self.out[key] = output[0] if isinstance(output, tuple) else output
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def response_token_index(attention_mask, response_mask):
    """Indices of the response tokens inside the remove-padding layout, in row-major order of response_mask."""
    batch, seqlen = attention_mask.shape
    resp_len = response_mask.shape[1]
    packed = attention_mask.flatten().cumsum(0) - 1  # padded flat position -> position among the unpadded tokens
    rows = torch.arange(batch, device=attention_mask.device).unsqueeze(1).expand(batch, resp_len)
    cols = torch.arange(seqlen - resp_len, seqlen, device=attention_mask.device).unsqueeze(0).expand(batch, resp_len)
    return packed[rows * seqlen + cols][response_mask.bool()]


def response_hidden(hidden, attention_mask, response_mask, remove_padding):
    """Rows of a module output that belong to response tokens, shape (n_response_tokens, d)."""
    if remove_padding:  # (1, total_nnz, d): flash-attn varlen layout without padding
        return hidden.reshape(-1, hidden.shape[-1])[response_token_index(attention_mask, response_mask)]
    return hidden[:, -response_mask.shape[1]:][response_mask.bool()]


class TameActor(DataParallelPPOActor):
    """Drop-in replacement of DataParallelPPOActor. Without gamma > 0 it behaves exactly like the parent."""

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        cfg = from_env()
        is_actor = actor_optimizer is not None  # the reference policy is built with the same class but no optimizer
        self.gamma = float(cfg["gamma"]) if is_actor else 0.0
        self._use_sae = self.gamma > 0
        self._single_minibatch = is_actor and cfg["method"] == "tame" and not cfg["no_feedback"]
        self._rec = None
        if self._use_sae:
            self._setup_sae(cfg)

    def _setup_sae(self, cfg):
        assert self.ulysses_sequence_parallel_size == 1, "the SAE penalty does not support Ulysses sequence parallelism"
        device = "cpu" if get_device_name() == "cpu" else f"{get_device_name()}:{get_device_id()}"
        feats = json.load(open(cfg["features"]))["targeted"]
        self.saes = load_saes(cfg["sae"], device=device, keys=set(feats))
        if cfg["random_features"]:
            from ..generate import random_features
            feats = random_features(feats, next(iter(self.saes.values())).W_enc.shape[0], cfg["seed"])
        self.feats = {k: torch.tensor(v, device=device) for k, v in feats.items()}
        self._rec = ActivationRecorder(self.actor_module, list(self.feats))

    def _adapter_disabled(self):
        if not hasattr(self.actor_module, "disable_adapter"):
            raise RuntimeError("TAME compares against the frozen pre-RL model, which needs LoRA: "
                               "set actor_rollout_ref.model.lora_rank > 0")
        return self.actor_module.disable_adapter()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, record=False):
        if not record:
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)
        self._rec.out.clear()
        self._rec.active = True
        try:
            outputs = super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)
        finally:
            self._rec.active = False
        attention_mask, response_mask = micro_batch["attention_mask"], micro_batch["response_mask"]
        outputs["tame_acts"] = {
            k: self.saes[k].encode(response_hidden(self._rec.out[k], attention_mask, response_mask,
                                                   self.use_remove_padding))[:, f]
            for k, f in self.feats.items()
        }
        self._rec.out.clear()
        return outputs

    def _base_activations(self, micro_batch, temperature):
        with torch.no_grad(), self._adapter_disabled():
            return self._forward_micro_batch(micro_batch, temperature, record=True)["tame_acts"]

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # TAME: the feedback loop adds second attempts and distilled rows, so the step batch is not a fixed multiple of the
        # mini-batch size. Update on the whole batch at once, as tame.train does.
        mini_batches = [data] if self._single_minibatch else data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = math.ceil(len(mini_batch) / self.config.ppo_micro_batch_size_per_gpu)  # TAME
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / len(mini_batch)  # TAME: real mini-batch size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # TAME: template-latent activations of the frozen pre-RL model (adapter disabled), then of the policy
                    base_acts = self._base_activations(model_inputs, temperature) if self._use_sae else None
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, record=self._use_sae
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self._use_sae:
                        # TAME: L_SAE = sum_i ReLU(a_i - a_i_base)^2 over the targeted template latents of every hooked module
                        sae_tok = sum((outputs["tame_acts"][k] - base_acts[k]).relu().pow(2).sum(-1) for k in base_acts)
                        sae_mat = torch.zeros(response_mask.shape, dtype=sae_tok.dtype, device=sae_tok.device)
                        sae_mat = sae_mat.masked_scatter(response_mask.bool(), sae_tok)
                        sae_loss = agg_loss(loss_mat=sae_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + self.gamma * sae_loss
                        micro_batch_metrics["actor/tame_sae_loss"] = sae_loss.detach().item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
