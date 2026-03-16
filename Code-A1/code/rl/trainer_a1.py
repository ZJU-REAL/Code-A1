import os
import uuid
from typing import Dict, List
import socket
import json

import torch
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from collections import defaultdict
import copy

from verl.trainer.ppo.ray_trainer import (
 RayPPOTrainer,
 ResourcePoolManager,
 compute_response_mask,
 compute_advantage,
 Role
)
from verl.trainer.ppo.metric_utils import (
 compute_data_metrics,
 compute_throughout_metrics,
 compute_timing_metrics,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import reduce_metrics
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.tracking import Tracking
from verl.utils.debug import marked_timer

from reward import extract_code_from_response

class MistakeBook:
	"""
	Mistake book manager for attack testcases and a blacklist.
	Structure:
	{
		question_id: {
			"attack_testcases": [
				{"testcase": str, "frequency": int}
			],
			"blacklisted": bool
		}
	}
	"""

	def __init__(self, mistake_book_dir, current_step):
		"""
		Initialize the mistake book manager.

		Args:
			mistake_book_dir: Directory for mistake book storage.
			current_step: Current training step. If > 0, load that step; if <= 0, load the latest.
		"""
		self.mistake_book = {}
		self.mistake_book_dir = mistake_book_dir
		self.load(current_step)

	def load(self, step):
		"""
		Load the mistake book from file.

		Args:
			step: Step to load. If > 0, load that step.
		"""
		mistake_book_path = os.path.join(self.mistake_book_dir, socket.gethostname(), f"mistake_book_step_{step}.json")
		if not os.path.exists(mistake_book_path):
			print(f"Mistake book file not found at step {step}: {mistake_book_path}")
			return

		try:
			with open(mistake_book_path, 'r') as f:
				self.mistake_book = json.load(f)
			print(f"Loaded mistake book from {mistake_book_path}")
		except Exception as e:
			raise Exception(f"Failed to load mistake book from {mistake_book_path}: {e}")

	def save(self, step):
		"""Save the mistake book to file."""
		host_dir = os.path.join(self.mistake_book_dir, socket.gethostname())
		os.makedirs(host_dir, exist_ok=True)

		mistake_book_path = os.path.join(host_dir, f"mistake_book_step_{step}.json")
		try:
			with open(mistake_book_path, 'w') as f:
				json.dump(self.mistake_book, f, indent=2)
			print(f"Saved mistake book to {mistake_book_path}")
		except Exception as e:
			print(f"Failed to save mistake book to {mistake_book_path}: {e}")

	def is_blacklisted(self, question_id):
		"""Check whether a question is blacklisted."""
		if question_id not in self.mistake_book:
			return False
		return self.mistake_book[question_id].get("blacklisted", False)

	def get_attack_testcases(self, question_id, max_count = 8):
		"""Get attack testcases for a question, sorted by frequency, up to max_count."""
		if question_id not in self.mistake_book:
			return []

		attack_testcases = self.mistake_book[question_id].get("attack_testcases", [])

		attack_testcases.sort(key=lambda x: x.get("frequency", 0), reverse=True)

		return [tc["testcase"] for tc in attack_testcases[:max_count]]

	def update(self, question_id: str, new_attack_testcases: List[str],
	           old_attack_testcases_passed: List[str], all_pass_rate: float):
		"""
		Update the mistake book.

		Args:
			question_id: Question ID.
			new_attack_testcases: Newly found attack testcases (strings, duplicates allowed).
			old_attack_testcases_passed: Passed old attack testcases (strings, duplicates allowed).
			all_pass_rate: Average pass rate across all code.
		"""
		if question_id not in self.mistake_book:
			self.mistake_book[question_id] = {
			 "attack_testcases": [],
			 "blacklisted": False
			}

		tc_freq_map = {}
		for tc_info in self.mistake_book[question_id]["attack_testcases"]:
			tc_freq_map[tc_info["testcase"]] = tc_info["frequency"]

		for tc in old_attack_testcases_passed:
			if tc in tc_freq_map:
				tc_freq_map[tc] -= 1

				if tc_freq_map[tc] <= 0:
					del tc_freq_map[tc]

		for tc in new_attack_testcases:
			if tc in tc_freq_map:
				tc_freq_map[tc] += 1
			else:
				tc_freq_map[tc] = 1

		self.mistake_book[question_id]["attack_testcases"] = [
		 {"testcase": tc, "frequency": freq}
		 for tc, freq in tc_freq_map.items()
		]

def _expand_batch(batch: DataProto, output_raw: DataProto, n_samples: int) -> DataProto:
	"""Expand batch"""
	batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

	batch = batch.repeat(repeat_times=n_samples, interleave=True)
	batch = batch.union(output_raw)

	batch.batch["response_mask"] = compute_response_mask(batch)

	batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
	return batch

def _prepare_old_log_prob(output, metrics, model_wg, config):
	"""Prepare old_log_prob"""
	old_log_prob = model_wg.compute_log_prob(output)
	entropys = old_log_prob.batch["entropys"]
	response_masks = output.batch["response_mask"]
	loss_agg_mode = config.actor_rollout_ref.actor.loss_agg_mode
	entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
	old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
	metrics.update(old_log_prob_metrics)
	old_log_prob.batch.pop("entropys")
	output = output.union(old_log_prob)
	return output, metrics

def _prepare_ref_log_prob(output, model_wg):
	"""Prepare ref_log_prob"""
	ref_log_prob = model_wg.compute_ref_log_prob(output)
	output = output.union(ref_log_prob)
	return output

def _save_checkpoint(config, model_wg, global_steps, dataloader_state_dict):

	local_global_step_folder = os.path.join(config.trainer.default_local_dir,
	socket.gethostname(),
	f"global_step_{global_steps}")
	local_global_step_folder = os.path.abspath(local_global_step_folder)

	print(f"local_global_step_folder: {local_global_step_folder}")
	actor_local_path = os.path.join(local_global_step_folder, "actor_rollout_ref")

	actor_remote_path = None

	remove_previous_ckpt_in_save = config.trainer.get("remove_previous_ckpt_in_save", False)
	if remove_previous_ckpt_in_save:
		print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
	max_actor_ckpt_to_keep = config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

	model_wg.save_checkpoint(actor_local_path, actor_remote_path, global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

	dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
	torch.save(dataloader_state_dict, dataloader_local_path)

	local_latest_checkpointed_iteration = os.path.join(config.trainer.default_local_dir, socket.gethostname(), "latest_checkpointed_iteration.txt")
	with open(local_latest_checkpointed_iteration, "w") as f:
		f.write(str(global_steps))

class ACGRayTrainer(RayPPOTrainer):
	"""
	Dual model collaborative trainer
	Inherits from RayPPOTrainer, extends support for dual model training simultaneously
	"""

	def __init__(
	 self,
	 config_a: DictConfig,
	 config_b: DictConfig,
	 tokenizer_a,
	 tokenizer_b,
	 role_worker_mapping: dict,
	 resource_pool_manager: ResourcePoolManager,
	 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
	 processor=None,
	 reward_fn=None,
	 val_reward_fn=None,
	 train_dataset=None,
	 val_dataset=None,
	 collate_fn=collate_fn,
	 train_sampler=None,
	 device_name="cuda",
	):
		super().__init__(
		 config=config_a,
		 tokenizer=tokenizer_a,
		 role_worker_mapping=role_worker_mapping,
		 resource_pool_manager=resource_pool_manager,
		 ray_worker_group_cls=ray_worker_group_cls,
		 processor=processor,
		 reward_fn=reward_fn,
		 val_reward_fn=val_reward_fn,
		 train_dataset=train_dataset,
		 val_dataset=val_dataset,
		 collate_fn=collate_fn,
		 train_sampler=train_sampler,
		 device_name=device_name,
		)

		self.config_a = config_a
		self.config_b = config_b
		self.tokenizer_a = tokenizer_a
		self.tokenizer_b = tokenizer_b
		self.mistake_book = None

	def init_workers(self):
		"""Initialize dual model workers"""
		print("Initializing dual model collaborative trainer...")

		self.resource_pool_manager.create_resource_pool()

		a_wg_kwargs = {
		 "ray_wait_register_center_timeout": self.config.trainer.get("ray_wait_register_center_timeout", None)
		}
		b_wg_kwargs = {
		 "ray_wait_register_center_timeout": self.config.trainer.get("ray_wait_register_center_timeout", None)
		}

		model_a_actor_rollout_cls = RayClassWithInitArgs(
		 cls=self.role_worker_mapping[Role.ActorRollout],
		 config=self.config.actor_rollout_ref,
		 role="actor_rollout",
		)

		model_a_ref_policy_cls = RayClassWithInitArgs(
		 cls=self.role_worker_mapping[Role.RefPolicy],
		 config=self.config.actor_rollout_ref,
		 role="ref",
		)
		worker_a_dict_cls = create_colocated_worker_cls(class_dict={
		 'actor_rollout': model_a_actor_rollout_cls,
		 'ref': model_a_ref_policy_cls,
		})
		a_wg_dict = self.ray_worker_group_cls(
		 resource_pool=self.resource_pool_manager.get_resource_pool(Role.ActorRollout),
		 ray_cls_with_init=worker_a_dict_cls,
		 device_name=self.device_name,
		 name_prefix="MA_Worker_Group",
		 **a_wg_kwargs
		)
		a_spawn_wg = a_wg_dict.spawn(prefix_set=["actor_rollout", "ref"])
		self.a_actor_rollout_wg = a_spawn_wg["actor_rollout"]
		self.a_ref_policy_wg = a_spawn_wg["ref"]

		try:
			OmegaConf.set_struct(self.config_b, True)
			with open_dict(self.config_b):
				self.config_b.actor_rollout_ref.actor.optim.total_training_steps = self.config.actor_rollout_ref.actor.optim.total_training_steps
		except Exception as e:
			raise Exception(f"Error: Could not set config. {e}")

		model_b_actor_rollout_cls = RayClassWithInitArgs(
		 cls=self.role_worker_mapping[Role.ActorRollout_B],
		 config=self.config_b.actor_rollout_ref,
		 role="actor_rollout",
		)

		model_b_ref_policy_cls = RayClassWithInitArgs(
		 cls=self.role_worker_mapping[Role.RefPolicy_B],
		 config=self.config_b.actor_rollout_ref,
		 role="ref",
		)
		worker_b_dict_cls = create_colocated_worker_cls(class_dict={
		 'actor_rollout': model_b_actor_rollout_cls,
		 'ref': model_b_ref_policy_cls,
		})
		b_wg_dict = self.ray_worker_group_cls(
		 resource_pool=self.resource_pool_manager.get_resource_pool(Role.ActorRollout_B),
		 ray_cls_with_init=worker_b_dict_cls,
		 device_name=self.device_name,
		 name_prefix="MB_Worker_Group",
		 **b_wg_kwargs
		)
		b_spawn_wg = b_wg_dict.spawn(prefix_set=["actor_rollout", "ref"])
		self.b_actor_rollout_wg = b_spawn_wg["actor_rollout"]
		self.b_ref_policy_wg = b_spawn_wg["ref"]

		@ray.remote
		def init_model_async(model_wg):
			return model_wg.init_model()

		ray.get([
		 init_model_async.remote(self.a_actor_rollout_wg),
		 init_model_async.remote(self.b_actor_rollout_wg)
		])
		ray.get([
		 init_model_async.remote(self.a_ref_policy_wg),
		 init_model_async.remote(self.b_ref_policy_wg)
		])
		print("Dual model workers initialized")

	def _dual_model_rollout(self, batch: DataProto) -> DataProto:
		"""
		Dual model rollout: simultaneously generate code and test cases
		Model A: generate code
		Model B: generate test cases
		"""
		print("Starting dual model rollout...")

		batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
		non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
		if "multi_modal_data" in batch.non_tensor_batch:
			non_tensor_batch_keys_to_pop.append("multi_modal_data")
		if "raw_prompt" in batch.non_tensor_batch:
			non_tensor_batch_keys_to_pop.append("raw_prompt")
		if "tools_kwargs" in batch.non_tensor_batch:
			non_tensor_batch_keys_to_pop.append("tools_kwargs")
		if "interaction_kwargs" in batch.non_tensor_batch:
			non_tensor_batch_keys_to_pop.append("interaction_kwargs")

		gen_batch_a = batch.pop(
		 batch_keys=batch_keys_to_pop,
		 non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
		)

		@ray.remote
		def generate_sequences_async(worker_group, input):
			return worker_group.generate_sequences(input)

		a_batch_future = generate_sequences_async.remote(self.a_actor_rollout_wg, gen_batch_a)
		print(f"Model A: call generate_sequences... (Input Batch size for A: {len(gen_batch_a.batch['input_ids'])})")

		print("Waiting for model A to complete...")
		gen_batch_output_a = ray.get(a_batch_future)
		n_samples_a = self.config_a.actor_rollout_ref.rollout.get("n", 1)
		print(f"Model A: generate_sequences completed. Output batch size: {len(gen_batch_output_a.batch['responses']) if gen_batch_output_a.batch is not None and 'responses' in gen_batch_output_a.batch else 'N/A'}")

		print("Model A: start decoding responses...")
		a_gen_code, generated_code_info = extract_code_from_response(self.tokenizer_a.batch_decode(gen_batch_output_a.batch["responses"], skip_special_tokens=True))
		print(f"Model A: decoding completed, {len(a_gen_code)} responses.")

		b2_batch = self._prepare_model_b_batch_phase2(batch, a_gen_code)
		print(f"Model B Phase 2: call generate_sequences... (Input Batch size for B2: {len(b2_batch.batch['input_ids'])})")

		gen_batch_b2 = b2_batch.pop(
		 batch_keys=batch_keys_to_pop,
		 non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
		)

		print("Asynchronously start model B Phase2 generation...")
		b2_batch_future = generate_sequences_async.remote(self.b_actor_rollout_wg, gen_batch_b2)

		a_batch = _expand_batch(batch, gen_batch_output_a, n_samples_a)
		a_batch.non_tensor_batch["generated_code"] = np.array(a_gen_code, dtype=object)

		print(f"Model A: expansion completed. Final batch size: {len(a_batch) if a_batch else 'N/A'}")

		n_samples_b = self.config_b.actor_rollout_ref.rollout.get("n", 1)

		gen_batch_output_b2 = ray.get(b2_batch_future)
		print(f"Model B Phase 2: generate_sequences completed. Output batch size: {len(gen_batch_output_b2.batch['responses']) if gen_batch_output_b2.batch is not None and 'responses' in gen_batch_output_b2.batch else 'N/A'}. gen_batch_output_b2.batch['responses']: {gen_batch_output_b2.batch['responses'].shape}")

		b2_batch = _expand_batch(b2_batch, gen_batch_output_b2, n_samples_b)

		print(f"Model B Phase 2: expansion completed. Final batch size: {len(b2_batch) if b2_batch else 'N/A'}")

		print("Dual model rollout completed, return independent outputs")
		return {
		 "original_batch": batch,
		 "a_batch": a_batch,
		 "b2_batch": b2_batch
		}

	def _prepare_model_b_batch_phase2(self, batch: DataProto, a_gen_code: List[str]) -> DataProto:
		"""
		Prepare model B Phase2: for each generation result of model A, generate difficult test cases.
		"""
		if not batch.non_tensor_batch:
			raise ValueError("batch.non_tensor_batch is empty")

		n_samples_a = self.config_a.actor_rollout_ref.rollout.get("n", 1)
		batch = batch.repeat(repeat_times=n_samples_a, interleave=True)
		if "clean_description" in batch.non_tensor_batch:
			a_bs = len(batch.non_tensor_batch["clean_description"])
		else:
			raise ValueError("cannot find 'clean_description' field in non_tensor_batch")

		tensors = defaultdict(list)
		non_tensors = defaultdict(list)
		for key, val in batch.non_tensor_batch.items():

			non_tensors[key].extend(val.tolist())

		for i in range(a_bs):
			clean_description = batch.non_tensor_batch.get("clean_description", [""]*a_bs)[i]
			generated_code = a_gen_code[i]
			prompt = [
			 {"role": "system", "content": "You are a helpful test case generation assistant."},
			 {"role": "user", "content": f"Given the following Question, generate 8 assert-based test cases to detect bugs in the following python function Buggy Code. Each test case must be in the format: 'assert function_name(parameters) == answer', and split by '\\n'. Output all assert statements inside '```python ... ```' code block, and do not output anything else.\n\nQuestion:\n{clean_description}\n\nBuggy Code:\n{generated_code}" if 'sft' in self.config_b.actor_rollout_ref.model.path else f"""
# Role
You are specializing in finding specific inputs that cause `Buggy Code` to behave differently from the requirements (`Question`).

# Task
Generate 8 assertion-based test cases to detect bugs for the function in `Buggy Code` according to the `Question`.

# Strategy
1. **Attack Logic Gaps**: Analyze where the `Buggy Code` logic might be "too simple" compared to the `Question`. Construct input `parameters` that hit these blind spots (e.g., missing constraints, misinterpreted rules, over-simplified logic).
2. **Prioritize Complexity**: Prefer complex input `parameters` (e.g., boundary values, nested loops, compound conditions, rare branches) over simple ones. Ensure every logical branch is stressed and every potential issue is covered.
3. **Zero Redundancy**: Do not brute-force generating trivial or repetitive tests. Only the first few generated tests will be evaluated, so quality and ordering matter more than quantity.

# Context
Question:
{clean_description}

Buggy Code:
```python
{generated_code}
```

# Output Format
Output ALL the assert statements inside ONE '```python ... ```' code block.
Format: 'assert function_name(parameters) == answer'
"""}

			]

			raw_prompt = self.tokenizer_b.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
			model_inputs = self.tokenizer_b(raw_prompt, return_tensors="pt", add_special_tokens=False)
			input_ids = model_inputs.pop("input_ids")
			attention_mask = model_inputs.pop("attention_mask")

			input_ids, attention_mask = verl_F.postprocess_data(
			 input_ids=input_ids,
			 attention_mask=attention_mask,
			 max_length=self.config_b.data.get("max_prompt_length", 2048),
			 pad_token_id=self.tokenizer_b.pad_token_id,
			 left_pad=True,
			 truncation=self.config_b.data.get("truncation", "error"),
			)

			position_ids = compute_position_id_with_mask(attention_mask)
			raw_prompt_ids = self.tokenizer_b.encode(raw_prompt, add_special_tokens=False)

			tensors["input_ids"].append(input_ids[0])
			tensors["attention_mask"].append(attention_mask[0])
			tensors["position_ids"].append(position_ids[0])

			non_tensors["a_idx"].append(i)
			non_tensors["generated_code"].append(generated_code)
			non_tensors["raw_prompt_ids"].append(raw_prompt_ids)
			non_tensors["tools_kwargs"].append(batch.non_tensor_batch.get("extra_info", [{}]*a_bs)[i].get("tools_kwargs", {}))
			non_tensors["interaction_kwargs"].append(batch.non_tensor_batch.get("extra_info", [{}]*a_bs)[i].get("interaction_kwargs", False))

		for key, val in tensors.items():
			tensors[key] = torch.stack(val, dim=0)

		for key, val in non_tensors.items():
			non_tensors[key] = np.array(val, dtype=object)

		return DataProto.from_single_dict({**tensors, **non_tensors})

	def _reconstruct_batch(self, batch):
		question_ids = [info.get("question_id", "") for info in batch.non_tensor_batch["extra_info"]]
		keep_indices = []
		for idx, qid in enumerate(question_ids):
			if not self.mistake_book.is_blacklisted(qid):
				keep_indices.append(idx)

		if len(keep_indices) < len(question_ids):
			print(f"Filtered {len(question_ids) - len(keep_indices)} blacklisted questions before rollout")
			indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
			batch.reorder(indices_tensor)
		return batch

	def _reconstruct_a_batch(self, a_batch, reward_a, reward_extra_infos_dict):
		"""
		Filter out questions with zero variance in rewards using vectorization.
		"""
		n_samples_a = self.config_a.actor_rollout_ref.rollout.get("n", 1)
		batch_size = len(a_batch) // n_samples_a

		reward_a_list = reward_extra_infos_dict.get("reward_a", [])
		rewards_tensor = torch.tensor(reward_a_list, device=reward_a.device)

		rewards_per_question = rewards_tensor.view(batch_size, -1)

		is_all_pass = (torch.abs(rewards_per_question - 1.0) < 1e-6).all(dim=1)
		is_all_fail = (torch.abs(rewards_per_question - 0.0) < 1e-6).all(dim=1)

		keep_mask_q = ~(is_all_pass | is_all_fail)

		keep_mask_samples = keep_mask_q.repeat_interleave(n_samples_a)

		indices_tensor = keep_mask_samples.nonzero(as_tuple=True)[0]

		if len(indices_tensor) < len(a_batch):
			print(f"Filtered {len(a_batch) - len(indices_tensor)} code samples (zero reward variance)")
			a_batch.reorder(indices_tensor)
			reward_a = reward_a[indices_tensor]

		return a_batch, reward_a, indices_tensor

	def _reconstruct_b2_batch(self, b2_batch, reward_b2, reward_extra_infos_dict, kept_a_indices=None):
		"""
		Reconstruct b2_batch by:
		1. Synchronizing with filtered a_batch (if a_batch was filtered) - reduce computation first
		2. Selecting testcases based on reward_b from the filtered items
		"""
		update_n = self.config_b.update_n
		reward_b_values = torch.tensor(reward_extra_infos_dict.get('reward_b', []), device=reward_b2.device)
		original_b2_len = len(reward_b_values)
		global_indices = torch.arange(original_b2_len, device=reward_b2.device)
		n_samples_a = self.config_a.actor_rollout_ref.rollout.get("n", 1)
		n_samples_b = self.config_b.actor_rollout_ref.rollout.get("n", 1)

		if kept_a_indices is not None and len(kept_a_indices) * n_samples_b < len(b2_batch):

			base_indices = kept_a_indices.repeat_interleave(n_samples_b) * n_samples_b

			offsets = torch.arange(n_samples_b, device=kept_a_indices.device).repeat(len(kept_a_indices))

			valid_b2_indices = base_indices + offsets

			print(f"Filtered {len(b2_batch) - len(valid_b2_indices)} test samples (sync with A)")
			reward_b_values = reward_b_values[valid_b2_indices]
			global_indices = global_indices[valid_b2_indices]

		current_total_samples = len(reward_b_values)

		current_batch_size = current_total_samples // (n_samples_a * n_samples_b)

		if current_total_samples > 0 and update_n < n_samples_a:

			rewards_view = reward_b_values.view(current_batch_size, n_samples_a, n_samples_b)

			std_per_code = torch.std(rewards_view.float(), dim=2, unbiased=False)

			_, top_code_indices = torch.topk(std_per_code, k=update_n, dim=1)

			batch_offsets = (torch.arange(current_batch_size, device=reward_b2.device) * n_samples_a * n_samples_b).unsqueeze(1)

			selected_code_starts = batch_offsets + (top_code_indices * n_samples_b)

			selected_code_starts_expanded = selected_code_starts.unsqueeze(2)

			tc_offsets = torch.arange(n_samples_b, device=reward_b2.device).view(1, 1, -1)

			final_indices = (selected_code_starts_expanded + tc_offsets).flatten()
			print(f"Selected {len(final_indices)} responses (Top {update_n} variance)")
			global_indices = global_indices[final_indices]

		b2_batch.reorder(global_indices)
		reward_b2 = reward_b2[global_indices]
		is_selected_mask = torch.zeros(original_b2_len, dtype=torch.bool, device=reward_b2.device)
		is_selected_mask[global_indices] = True
		reward_extra_infos_dict['is_selected'] = is_selected_mask.tolist()

		return b2_batch, reward_b2, reward_extra_infos_dict,

	def _dual_model_training_step(self, batch, a_metrics, b2_metrics, timing_raw):
		"""Dual model training step"""

		print("Start dual model rollout...")
		with marked_timer("gen", timing_raw):
			rollout_results = self._dual_model_rollout(batch)

		original_batch = rollout_results["original_batch"]
		a_batch = rollout_results["a_batch"]
		b2_batch = rollout_results["b2_batch"]

		print("Compute dual model collaborative rewards...")
		@ray.remote
		def compute_reward_async(batch_1, batch_2, reward_fn):
			return reward_fn(batch_1, batch_2)

		with marked_timer("reward", timing_raw):

			reward_result = self.reward_fn(a_batch, b2_batch)
			reward_a = reward_result["reward_a_tensor"]
			reward_b2 = reward_result["reward_b2_tensor"]
			reward_extra_infos_dict = reward_result["reward_extra_infos_dict"]
			sandbox_infos_dict = reward_result["sandbox_infos_dict"]
			sandbox_metrics = reward_result["sandbox_metrics"]
			a_metrics.update(sandbox_metrics)
			a_metrics.update({"reward/mean": reward_a.sum(-1).mean().item(), "reward/std": reward_a.sum(-1).std().item()})
			b2_metrics.update({"reward/mean": reward_b2.sum(-1).mean().item(), "reward/std": reward_b2.sum(-1).std().item()})

		all_a_batch = copy.deepcopy(a_batch)
		all_reward_a = copy.deepcopy(reward_a)

		all_b2_batch = copy.deepcopy(b2_batch)
		all_reward_b2 = copy.deepcopy(reward_b2)

		b2_batch, reward_b2, reward_extra_infos_dict = self._reconstruct_b2_batch(b2_batch, reward_b2, reward_extra_infos_dict, None)
		b2_metrics.update({"reward/selected_mean": reward_b2.sum(-1).mean().item(), "reward/selected_std": reward_b2.sum(-1).std().item()})

		print("Prepare log_prob...")

		@ray.remote
		def _prepare_old_log_prob_async(batch, metrics, model_wg, config):
			return _prepare_old_log_prob(batch, metrics, model_wg, config)

		with marked_timer("old_log_prob", timing_raw):

			a_batch_future = _prepare_old_log_prob_async.remote(a_batch, a_metrics, self.a_actor_rollout_wg, self.config)
			b2_batch_future = _prepare_old_log_prob_async.remote(b2_batch, b2_metrics, self.b_actor_rollout_wg, self.config)

			a_batch, a_metrics = ray.get(a_batch_future)
			b2_batch, b2_metrics = ray.get(b2_batch_future)

		@ray.remote
		def _prepare_ref_log_prob_async(batch, model_wg):
			return _prepare_ref_log_prob(batch, model_wg)

		with marked_timer("ref", timing_raw):

			a_batch_future = _prepare_ref_log_prob_async.remote(a_batch, self.a_ref_policy_wg)
			b2_batch_future = _prepare_ref_log_prob_async.remote(b2_batch, self.b_ref_policy_wg)

			a_batch = ray.get(a_batch_future)
			b2_batch = ray.get(b2_batch_future)

		print("Compute token-level rewards and advantages...")

		with marked_timer("adv", timing_raw):
			a_batch.batch["token_level_scores"] = reward_a
			b2_batch.batch["token_level_scores"] = reward_b2

			a_batch.batch["token_level_rewards"] = a_batch.batch["token_level_scores"]
			b2_batch.batch["token_level_rewards"] = b2_batch.batch["token_level_scores"]

			a_batch = compute_advantage(
			 a_batch,
			 adv_estimator=self.config.algorithm.adv_estimator,
			 norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
			)
			b2_batch = compute_advantage(
			 b2_batch,
			 adv_estimator=self.config.algorithm.adv_estimator,
			 norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
			)

		print("Asynchronously update actor...")

		@ray.remote
		def update_actor_async(worker_group, batch):
			return worker_group.update_actor(batch)

		with marked_timer("update_actor", timing_raw):

			a_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
			b2_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable

			a_batch_future = update_actor_async.remote(self.a_actor_rollout_wg, a_batch)
			b2_batch_future = update_actor_async.remote(self.b_actor_rollout_wg, b2_batch)

			a_actor_output = ray.get(a_batch_future)
			b2_actor_output = ray.get(b2_batch_future)

			model_a_metrics = reduce_metrics(a_actor_output.meta_info["metrics"])
			model_b_phase2_metrics = reduce_metrics(b2_actor_output.meta_info["metrics"])

			a_metrics.update(model_a_metrics)
			b2_metrics.update(model_b_phase2_metrics)

		a_rollout_data_dir = self.config_a.trainer.get("rollout_data_dir", None)
		if a_rollout_data_dir:
				a_rollout_data_dir = os.path.join(a_rollout_data_dir, socket.gethostname())
				inputs = self.tokenizer_a.batch_decode(all_a_batch.batch["prompts"], skip_special_tokens=True)
				outputs = self.tokenizer_a.batch_decode(all_a_batch.batch["responses"], skip_special_tokens=True)
				scores = all_reward_a.sum(-1).cpu().tolist()
				self._dump_generations(
				 inputs=inputs,
				 outputs=outputs,
				 scores=scores,
				 reward_extra_infos_dict=sandbox_infos_dict,
				 dump_path=a_rollout_data_dir,
				)

		b_rollout_data_dir = self.config_b.trainer.get("rollout_data_dir", None)
		if b_rollout_data_dir:
				b_rollout_data_dir = os.path.join(b_rollout_data_dir, socket.gethostname())
				b2_inputs = self.tokenizer_b.batch_decode(all_b2_batch.batch["prompts"], skip_special_tokens=True)
				b2_outputs = self.tokenizer_b.batch_decode(all_b2_batch.batch["responses"], skip_special_tokens=True)
				scores = all_reward_b2.sum(-1).cpu().tolist()
				self._dump_generations(
				 inputs=b2_inputs,
				 outputs=b2_outputs,
				 scores=scores,
				 reward_extra_infos_dict=reward_extra_infos_dict,
				 dump_path=b_rollout_data_dir + "_phase2",
				)

		if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (self.is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
			with marked_timer("testing", timing_raw):
				val_metrics: dict = self._validate()
				if self.is_last_step:
					print(f"Final validation metrics: {val_metrics}")
			a_metrics.update(val_metrics)

		@ray.remote
		def _save_checkpoint_async(config, model_wg, global_steps, dataloader_state_dict):
			return _save_checkpoint(config, model_wg, global_steps, dataloader_state_dict)

		if self.config.trainer.save_freq > 0 and (self.is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
			with marked_timer("save_checkpoint", timing_raw):
				ray.get([
				 _save_checkpoint_async.remote(self.config_a, self.a_actor_rollout_wg, self.global_steps, self.train_dataloader.state_dict()),
				 _save_checkpoint_async.remote(self.config_b, self.b_actor_rollout_wg, self.global_steps, self.train_dataloader.state_dict()),
				])

		if self.mistake_book:
			self.mistake_book.save(self.global_steps)

		print("Dual model training step completed")
		return a_batch, b2_batch, a_metrics, b2_metrics, timing_raw

	def _load_checkpoint(self, config, model_wg):
		if config.trainer.resume_mode == "disable":
			return 0

		checkpoint_folder = os.path.join(config.trainer.default_local_dir, socket.gethostname())
		if not os.path.isabs(checkpoint_folder):
			checkpoint_folder = os.path.abspath(checkpoint_folder)
		global_step_folder = find_latest_ckpt_path(checkpoint_folder)

		if config.trainer.resume_mode == "auto":
			if global_step_folder is None:
				print("Training from scratch")
				return 0
		else:
			raise NotImplementedError("resume mode not implemented")

		print(f"Load from checkpoint folder: {global_step_folder}")

		self.global_steps = int(global_step_folder.split("global_step_")[-1])

		print(f"Setting global step to {self.global_steps}")
		print(f"Resuming from {global_step_folder}")

		actor_rollout_ref_path = os.path.join(global_step_folder, "actor_rollout_ref")

		model_wg.load_checkpoint(actor_rollout_ref_path, del_local_after_load=config.trainer.del_local_ckpt_after_load)

		dataloader_local_path = os.path.join(global_step_folder, "data.pt")
		if os.path.exists(dataloader_local_path):
			dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
			self.train_dataloader.load_state_dict(dataloader_state_dict)
		else:
			print(f"No dataloader state found at {dataloader_local_path}, will start from scratch")

	def collect_metrics(self, batch, metrics, timing_raw, group=None):

		metrics.update(
		 {
		  "training/global_step": self.global_steps,
		  "training/epoch": self.global_epochs,
		 }
		)

		metrics.update(compute_data_metrics(batch=batch, use_critic=False))
		metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
		n_gpus = self.resource_pool_manager.get_n_gpus()
		metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

		if group:
			metrics = {
			 (f"{k.split('/', 1)[0]}_{group}/{k.split('/', 1)[1]}" if "/" in k else f"{k}_{group}"): v
			 for k, v in metrics.items()
			}

		return metrics

	def fit(self):
		"""Dual model collaborative training main loop"""
		print("Start dual model collaborative training...")

		logger = Tracking(
		 project_name=self.config.trainer.project_name,
		 experiment_name=self.config.trainer.experiment_name,
		 default_backend=self.config.trainer.logger,
		 config=OmegaConf.to_container(self.config, resolve=True),
		)

		self.global_steps = 0

		self._load_checkpoint(self.config_a, self.a_actor_rollout_wg)
		self._load_checkpoint(self.config_b, self.b_actor_rollout_wg)

		if self.config_a.get("mistake_book", None):
			mistake_book_dir = self.config_a.trainer.get("rollout_data_dir", None)
			self.mistake_book = MistakeBook(mistake_book_dir, self.global_steps)
			self.reward_fn.mistake_book = self.mistake_book

		self.is_last_step = self.global_steps >= self.total_training_steps
		if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
			val_metrics = self._validate()
			assert val_metrics, f"{val_metrics=}"
			a_metrics = {
			 (f"{k.split('/', 1)[0]}_a/{k.split('/', 1)[1]}" if "/" in k else f"{k}_a"): v
			 for k, v in val_metrics.items()
			}
			print(f"Initial validation metrics: {a_metrics}")
			logger.log(data=a_metrics, step=self.global_steps)
			if self.config.trainer.get("val_only", False):
				if logger and hasattr(logger, "finish"):
					logger.finish()
					import time
					time.sleep(10)
				return

		if self.is_last_step:
			print("Training finished, no need to start training")
			return

		progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

		self.global_steps += 1

		print(f"\n{'='*60}")
		print("Configuration validation:")
		print(f"  - Dataset size: {len(self.train_dataset)}")
		print(f"  - Dataloader batch size: {len(self.train_dataloader)}")
		print(f"{'='*60}")

		last_batch = None
		for epoch in range(self.config.trainer.total_epochs):
			self.global_epochs = epoch + 1
			for batch_dict in self.train_dataloader:
				a_metrics = {}
				b2_metrics = {}
				timing_raw = {}
				batch: DataProto = DataProto.from_single_dict(batch_dict)

				self.is_last_step = self.global_steps >= self.total_training_steps

				with marked_timer("step", timing_raw):
					a_batch, b2_batch, a_metrics, b2_metrics, timing_raw = self._dual_model_training_step(batch, a_metrics, b2_metrics, timing_raw)

				a_metrics = self.collect_metrics(a_batch, a_metrics, timing_raw, group="a")
				b2_metrics = self.collect_metrics(b2_batch, b2_metrics, timing_raw, group="b2")

				logger.log(data=a_metrics, step=self.global_steps)
				logger.log(data=b2_metrics, step=self.global_steps)

				if self.is_last_step:
					progress_bar.close()
					if logger and hasattr(logger, "finish"):
						logger.finish()
						import time
						time.sleep(10)
					return

				progress_bar.update(1)
				self.global_steps += 1

	def _validate(self):
		data_source_lst = []
		uid_lst = []

		sample_inputs = []
		sample_outputs = []
		sample_scores = []

		for test_data in self.val_dataloader:
			test_batch = DataProto.from_single_dict(test_data)

			test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
			test_batch = test_batch.repeat(repeat_times=self.config_a.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

			input_ids = test_batch.batch["input_ids"]
			input_texts = [self.tokenizer_a.decode(ids, skip_special_tokens=True) for ids in input_ids]
			sample_inputs.extend(input_texts)

			batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
			non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
			if "multi_modal_data" in test_batch.non_tensor_batch:
				non_tensor_batch_keys_to_pop.append("multi_modal_data")
			if "raw_prompt" in test_batch.non_tensor_batch:
				non_tensor_batch_keys_to_pop.append("raw_prompt")
			if "tools_kwargs" in test_batch.non_tensor_batch:
				non_tensor_batch_keys_to_pop.append("tools_kwargs")
			if "interaction_kwargs" in test_batch.non_tensor_batch:
				non_tensor_batch_keys_to_pop.append("interaction_kwargs")
			test_gen_batch = test_batch.pop(
			 batch_keys=batch_keys_to_pop,
			 non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
			)

			test_gen_batch.meta_info = {
			 "eos_token_id": self.tokenizer_a.eos_token_id,
			 "pad_token_id": self.tokenizer_a.pad_token_id,
			 "recompute_log_prob": False,
			 "do_sample": self.config_a.actor_rollout_ref.rollout.val_kwargs.do_sample,
			 "validate": True,
			}
			print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

			test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.a_actor_rollout_wg.world_size)
			test_output_gen_batch_padded = self.a_actor_rollout_wg.generate_sequences(test_gen_batch_padded)

			test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
			print("validation generation end")

			output_ids = test_output_gen_batch.batch["responses"]
			output_texts = [self.tokenizer_a.decode(ids, skip_special_tokens=True) for ids in output_ids]
			sample_outputs.extend(output_texts)

			test_batch = test_batch.union(test_output_gen_batch)

			reward_result = self.val_reward_fn(test_batch, is_last_step = self.is_last_step)
			scores = reward_result["pass_results"]
			sandbox_infos_dict = reward_result["sandbox_infos_dict"]
			sandbox_metrics = reward_result["sandbox_metrics"]
			sample_scores.extend(scores)

			data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores)))
			uid_lst.append(test_batch.non_tensor_batch.get("uid", ["unknown"] * len(scores)))

		a_val_data_dir = self.config_a.trainer.get("validation_data_dir", None)
		if a_val_data_dir:
			a_val_data_dir = os.path.join(a_val_data_dir, socket.gethostname())
			self._dump_generations(
			 inputs=sample_inputs,
			 outputs=sample_outputs,
			 scores=sample_scores,
			 reward_extra_infos_dict=sandbox_infos_dict,
			 dump_path=a_val_data_dir,
			)

		data_sources = np.concatenate(data_source_lst, axis=0)
		uids = np.concatenate(uid_lst, axis=0)

		metric_dict = process_validation_metrics(data_sources, uids, sample_scores)
		metric_dict.update(sandbox_metrics)

		return metric_dict

def estimate_pass_at_k(
 num_samples: np.ndarray,
 num_correct: np.ndarray,
 k: int,
) -> np.ndarray:
	"""
	Estimates pass@k of each problem and returns them in an array.
	"""

	def estimator(n: int, c: int, k: int) -> float:
		"""
		Calculates 1 - comb(n - c, k) / comb(n, k).
		"""
		if n - c < k:
			return 1.0
		return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

	assert len(num_samples) == len(num_correct)
	num_samples_it = iter(num_samples)

	return np.array(
	 [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
	)

def estimate_avg_at_k(
 raw_correct: np.ndarray,
 k: int,
) -> np.ndarray:
	if k == 0:
		return np.zeros(len(raw_correct))
	return raw_correct[:, :k].sum(axis=1) / k

def process_validation_metrics(
 data_sources: np.ndarray,
 uids: np.ndarray,
 pass_results: List[bool]
) -> Dict[str, float]:
	"""
	Process validation metrics for pass@k and avg@k statistics.

	Args:
		data_sources: List of data source identifiers for each sample
		uids: List of unique identifiers for each sample
		pass_results: List of boolean pass results for each sample

	Returns:
		Dictionary of metrics with keys like "val/{data_source}/pass@k"
	"""
	df = pd.DataFrame({
	 'data_source': data_sources,
	 'uid': uids,
	 'pass': pass_results
	})

	metric_dict = {}

	for data_source, group_df in df.groupby('data_source'):
		agg_results = group_df.groupby('uid')['pass'].agg(
		 total='count',
		 correct='sum'
		).reset_index()

		total = agg_results['total'].to_numpy()
		correct = agg_results['correct'].to_numpy()
		raw_correct = np.array(group_df.groupby('uid')['pass'].apply(list).tolist())

		ks = [1, 8, 16, 32, 64]

		for k in ks:
			if k <= total.min():
				pass_at_k = estimate_pass_at_k(total, correct, k).mean()
				avg_at_k = estimate_avg_at_k(raw_correct, k).mean()

				metric_key = f"val/{data_source}/pass@{k}"
				metric_dict[metric_key] = float(pass_at_k)
				metric_key = f"val/{data_source}/avg@{k}"
				metric_dict[metric_key] = float(avg_at_k)
	return metric_dict
