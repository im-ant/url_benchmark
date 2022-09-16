# ============================================================================
# Helper executor for running hydra based jobs on SLURM
#
# Author: AC
# ============================================================================


from datetime import datetime
import itertools
from pathlib import Path
import random
import string
import subprocess


class SbatchExecutor:
    def __init__(self, folder, mem_gb=32, n_cpus=8, gres=None, time='08:00:00'):
        self.folder = Path(folder)

        self.mem_gb = mem_gb
        self.n_cpus = n_cpus
        self.gres = gres
        self.time = time

        self.hydra_multirun = True

        self.args_dict = None
        self.args_grid = None

    def update_arguments(self, args_dict):
        """
        Get a dictionary of arguments for hydra. Turn into grid of arguments
        :param args_dict:

        Example: args_dict = {
            'agent': 'np_proto_actor', 'hydra.launcher.n_jobs': 3,
            'task': ['walker_stand', 'walker_walk'],
            'agent.lr': ['1e-4','1e-3','2e-3'],
            'seed': '1,2,3',
            'device': 'cuda',
        }
        """
        # Make the non-list items one-item list so it is iterable
        for k in args_dict:
            if type(args_dict[k]) != list:
                args_dict[k] = [args_dict[k]]

        # Grid outer product
        args_grid = list(
            dict(zip(args_dict.keys(), values)) for values in itertools.product(*args_dict.values())
        )

        self.args_dict = args_dict
        self.args_grid = args_grid

    def _process_hydra_arg(self, h_arg):
        """
        Helper method to escape out of character in the arguments to write
        the bash script
        :param h_arg:
        :return:
        """
        if type(h_arg) == str:
            h_arg = h_arg.replace('"', r'\"')
            h_arg = h_arg.replace('$', r'\\\$')
            # h_arg = h_arg.replace('%', r'\%')

        return h_arg

    def _write_slurm_file(self, dir_path, global_jobid, job_idx, arg_dict):
        # SLURM resource requests
        # TODO: need to auto-terminate if error, e.g. if there are no devices found
        #       when running on cpu only with device set to GPU. figure out bug here.
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={global_jobid}_{job_idx}',
            '#SBATCH --open-mode=append',
            f'#SBATCH --output={dir_path}/%j_%x.out',
            f'#SBATCH --error={dir_path}/%j_%x.err',
            '#SBATCH --export=ALL',
        ]

        if self.gres is not None:
            lines.extend([f'#SBATCH --gres={self.gres}'])

        lines.extend([
            f'#SBATCH --time={self.time}',
            f'#SBATCH --mem={self.mem_gb}G',
            f'#SBATCH --cpus-per-task={self.n_cpus}',
            '',
            '',
        ])

        # Start singularity container
        lines.extend([
            'singularity exec --nv --bind /share/apps \\',
            '\t\t--overlay $SCRATCH/urlb-overlay.ext3:ro \\',
            '\t\t/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c \"',
            '',
            'source /ext3/miniconda3/etc/profile.d/conda.sh',
            'export PATH=/ext3/miniconda3/bin:$PATH',
            'conda activate /ext3/urlb',
            '',
            '',
        ])

        # Python job
        # TODO: set this script to be customizable
        script_path = '/home/agc9824/Project/url_benchmark/finetune.py'
        lines.append(f'/ext3/urlb/bin/python {script_path} \\')

        # Write arguments
        for k in arg_dict:
            arg_value = self._process_hydra_arg(arg_dict[k])
            lines.append(f'\t{k}={arg_value} \\')

        if self.hydra_multirun:
            lines.append(f'\thydra.sweep.dir={dir_path} \\')
            lines.append('\thydra.sweep.subdir=\\$\\{hydra.job.num\\} \\')
            lines.append(f'\t--multirun')
        else:
            # TODO: add single run stuff
            lines.append('')

        # Finish
        lines.extend(['', '\"', ''])

        # Write
        filepath = dir_path / 'job.slurm'
        with open(filepath, 'w') as file:
            for line in lines:
                file.write(f'{line}\n')

        return filepath


    def submit(self):
        # Get current datetime
        now = datetime.today()
        now_date = now.strftime('%Y.%m.%d')
        now_time = now.strftime('%H%M%S')

        # Create global job name
        rand_str = ''.join(random.choices(
            string.ascii_letters + string.digits, k=4))
        global_jobid = f'{now_time}_{rand_str}'

        # Create parent directory
        exp_dir_path = self.folder / now_date / global_jobid

        exp_dir_path.mkdir(parents=True, exist_ok=False)
        print(f'Job folder: {exp_dir_path}')

        # Iterate through each argument configuration
        for grid_idx, cur_arg in enumerate(self.args_grid):
            # Create sub-directory
            sub_dir_path = exp_dir_path / str(grid_idx)
            sub_dir_path.mkdir(parents=False, exist_ok=False)

            # Write bash file
            slurm_filepath = self._write_slurm_file(
                sub_dir_path, global_jobid, grid_idx, cur_arg)

            # Submit job
            sb_out = subprocess.run(['sbatch', str(slurm_filepath)],
                                    stdout=subprocess.PIPE)
            print(sb_out)  # ??

        print(f'Date time: {now_date}-{now_time}; job custom id: {global_jobid}')


