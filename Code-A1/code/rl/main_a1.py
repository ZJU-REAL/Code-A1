import os
import sys
import ray
from pathlib import Path
import time
import torch
torch.compiler.reset()

from trainer_a1 import ACGRayTrainer

def main():
	print("Start dual model collaborative training...")

	config_a_path = sys.argv[1] if len(sys.argv) > 1 else "./config_model_a.yaml"

	from omegaconf import OmegaConf, open_dict
	config_a = OmegaConf.load(config_a_path)

	try:
		OmegaConf.set_struct(config_a, True)
		with open_dict(config_a):
			config_a.prefix = os.environ.get('PREFIX', str(Path.cwd()))
	except Exception as e:
		raise Exception(f"Error: Could not set prefix. {e}")

	OmegaConf.resolve(config_a)

	import copy
	config_b = copy.deepcopy(config_a)
	try:
		OmegaConf.set_struct(config_b, True)
		with open_dict(config_b):
			if OmegaConf.select(config_b, "other.data.max_prompt_length"):
				config_b.data.max_prompt_length = config_b.other.data.max_prompt_length
			config_b.actor_rollout_ref.model.path = config_b.other.actor_rollout_ref.model.path
			if OmegaConf.select(config_b, "other.actor_rollout_ref.rollout.n"):
				config_b.actor_rollout_ref.rollout.n = config_b.other.actor_rollout_ref.rollout.n
			config_b.trainer.rollout_data_dir = config_b.other.trainer.rollout_data_dir
			config_b.trainer.validation_data_dir = config_b.other.trainer.validation_data_dir
			if OmegaConf.select(config_b, "other.trainer.n_gpus_per_node"):
				config_b.trainer.n_gpus_per_node = config_b.other.trainer.n_gpus_per_node
			config_b.trainer.default_local_dir = config_b.other.trainer.default_local_dir
			config_b.update_n = config_b.actor_rollout_ref.rollout.n
			if OmegaConf.select(config_b, "other.update_n"):
				config_b.update_n = config_b.other.update_n
	except Exception as e:
		raise Exception(f"Error: Could not set config. {e}")

	if not ray.is_initialized():
		ray.init(runtime_env={'excludes': ['checkpoints', 'logs', 'bcb_results']})

	runner = TaskRunner.remote()
	ray.get(runner.run.remote(config_a, config_b))

	print("Dual model collaborative training completed!")

@ray.remote(num_cpus=1)
class TaskRunner:
	def run(self, config_a, config_b):
		from verl.utils import hf_tokenizer

		trust_remote_code = config_a.data.get("trust_remote_code", True)
		tokenizer_a = hf_tokenizer(config_a.actor_rollout_ref.model.path, trust_remote_code=trust_remote_code)
		tokenizer_b = hf_tokenizer(config_b.actor_rollout_ref.model.path, trust_remote_code=trust_remote_code)

		if config_a.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
			from verl.single_controller.ray import RayWorkerGroup
			from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

			actor_rollout_cls = AsyncActorRolloutRefWorker if config_a.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
			ray_worker_group_cls = RayWorkerGroup
		else:
			raise NotImplementedError

		from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

		role_worker_mapping = {
		 Role.ActorRollout: ray.remote(actor_rollout_cls),
		 Role.ActorRollout_B: ray.remote(actor_rollout_cls),
		}

		model_a_pool_id = "model_a_pool"
		model_b_pool_id = "model_b_pool"

		resource_pool_spec = {
		 model_a_pool_id: [config_a.trainer.n_gpus_per_node] * config_a.trainer.nnodes,
		 model_b_pool_id: [config_b.trainer.n_gpus_per_node] * config_b.trainer.nnodes,
		}

		mapping = {
		 Role.ActorRollout: model_a_pool_id,
		 Role.ActorRollout_B: model_b_pool_id,
		}

		if config_a.algorithm.use_kl_in_reward or config_a.actor_rollout_ref.actor.use_kl_loss:
			role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
			mapping[Role.RefPolicy] = model_a_pool_id

		if config_b.algorithm.use_kl_in_reward or config_b.actor_rollout_ref.actor.use_kl_loss:
			role_worker_mapping[Role.RefPolicy_B] = ray.remote(ActorRolloutRefWorker)
			mapping[Role.RefPolicy_B] = model_b_pool_id

		from verl.trainer.ppo.reward import load_reward_manager
		reward_fn = load_reward_manager(config_a, {tokenizer_a, tokenizer_b}, num_examine=0, **config_a.reward_model.get("reward_kwargs", {}))
		val_reward_fn = load_reward_manager(config_a, {tokenizer_a, tokenizer_b}, num_examine=1, **config_a.reward_model.get("reward_kwargs", {}))

		resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

		from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

		train_dataset = create_rl_dataset(config_a.data.train_files, config_a.data, tokenizer_a, None)
		val_dataset = create_rl_dataset(config_a.data.val_files, config_a.data, tokenizer_a, None)
		train_sampler = create_rl_sampler(config_a.data, train_dataset)

		trainer = ACGRayTrainer(
		 config_a=config_a,
		 config_b=config_b,
		 tokenizer_a=tokenizer_a,
		 tokenizer_b=tokenizer_b,
		 role_worker_mapping=role_worker_mapping,
		 resource_pool_manager=resource_pool_manager,
		 ray_worker_group_cls=ray_worker_group_cls,
		 reward_fn=reward_fn,
		 val_reward_fn=val_reward_fn,
		 train_dataset=train_dataset,
		 val_dataset=val_dataset,
		 train_sampler=train_sampler
		)
		trainer.init_workers()
		trainer.fit()

if __name__ == "__main__":
	main()
