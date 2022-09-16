# ============================================================================
# Run with:
#   . python-greene submit_slurm_sweep.py
#
# Design philosophy
#
# Author: AC
# ============================================================================

from slurm_helper import SbatchExecutor

class JobConfig:
    def __init__(self):
        pass


ArgsDict = {
    'agent': 'np_proto_actor',
    'hydra.launcher.n_jobs': 3,
    'experiment': 'finetune', 'use_tb': True,
    'task': ['walker_stand', 'walker_walk', 'walker_run', 'walker_flip'],
    'obs_type': 'pixels', 'action_repeat': 2,
    'snapshot_base_dir': '/scratch/agc9824/project_outputs/url_benchmark/',
    'snapshot_child_dir': '2022.08.16/142819_proto/pixels/walker/1',
    'snapshot_ts': 2000000, 'save_snapshots': False,
    'num_train_frames': 200010,
    'agent.name': 'np_actor_proto',
    'agent.init_actor': True, 'agent.init_critic': True,
    'agent.grad_critic_params': 'null', 'update_encoder': [False],
    'agent.hidden_dim': 256, 'agent.batch_size': 1024,
    'agent.lr': ['1e-4'],
    'agent.critic_target_tau': '1e-2',
    'agent.stddev_schedule': 0.2,
    'agent.actor_cfg.snd_kwargs.temperature': [0.1],
    'agent.actor_cfg.snd_kwargs.keys_grad': False,
    'agent.actor_cfg.snd_kwargs.values_grad': True,
    'agent.actor_cfg.snd_kwargs.temperature_grad': False,
    'seed': '1,2,3',
    'device': 'cuda',
}


def main():
    print('hello world')

    # For two jobs on lambda machine: starts at 18G, goes to 30+G

    folder = '/scratch/agc9824/project_outputs/url_benchmark/'
    executor = SbatchExecutor(folder, mem_gb=30, n_cpus=12, gres='gpu:1', time='8:00:00')

    executor.update_arguments(ArgsDict)
    # print(len(executor.args_grid))

    executor.submit()





if __name__ == "__main__":
    main()