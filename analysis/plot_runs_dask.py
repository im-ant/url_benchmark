#

from pathlib import Path
import yaml

import dask
import dask.bag as db
import dask.dataframe as dd
from dask.distributed import Client, progress
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_yaml(path):
    """
    Helper method: read a yaml file into a nested dictionary
    :param path:
    :return:
    """
    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            yaml_dict = None
        return yaml_dict


def dict2df(in_dict, prefix=None):
    """
    Helper method: take a (nested) dictionary and turn it into a 1-row long dataframe
    :param in_dict:
    :param prefix: None, or string prefix to add to start of the dataframe column names
    :return:
    """

    def unnest_dict(nest_dict, flat_dict, prefix=None):
        """
        Helper recursive method to un-nest a dictionary
        :param nest_dict:
        :param flat_dict:
        :param prefix:
        :return:
        """
        for k in nest_dict:
            # Base case: at leaf nodes, not more dicts
            if type(nest_dict[k]) is not dict:
                if prefix is None:
                    flat_k = k
                else:
                    flat_k = f'{prefix}.{k}'
                flat_v = str(nest_dict[k]) if isinstance(nest_dict[k], list) else nest_dict[k]
                flat_dict[flat_k] = flat_v
            # Recursion
            else:
                if prefix is None:
                    new_prefix = k
                else:
                    new_prefix = f'{prefix}.{k}'
                unnest_dict(nest_dict[k], flat_dict, new_prefix)
        return flat_dict

    # Get an un-nested dictionary from the hydra config yaml file
    config_dict = unnest_dict(in_dict, {}, prefix=prefix)

    # Make the dictionary into a 1-row dataframe
    predf_dict = {}
    for k in config_dict:
        predf_dict[k] = [config_dict[k], ]
    config_df = pd.DataFrame(predf_dict)

    return config_df


def load_configs(train_files):
    """
    Given a list of paths pointing to training files, get their associated .yaml config files
    and concatenate into a single pandas DataFrame
    :param train_files:
    :return:
    """
    configs_paths = [cur_p.parents[0] / '.hydra/config.yaml' for cur_p in train_files]
    config_dfs_bag = db.from_sequence(
        configs_paths, npartitions=(len(configs_paths)//100 + 1)  # 100 files per partition
    ).map(lambda p: dict2df(read_yaml(p), prefix=None))

    print('Processing config files:', config_dfs_bag)
    configs_df = pd.concat(config_dfs_bag.compute())
    return configs_df


def describe_configs_df(configs_df):
    print(configs_df)

    single_item_cols = {}
    multi_items_cols = {}
    for col in configs_df:
        items_list = list(configs_df[col].unique())
        if len(items_list) > 1:
            multi_items_cols[col] = items_list
        else:
            single_item_cols[col] = items_list

    def helper_print(items_dict):
        for k in items_dict:
            print(k)
            for ele in items_dict[k]:
                print('  ', ele)
    print('=== Independent Variables ===')
    helper_print(multi_items_cols)


def get_filtered_runs_paths(train_files, filter_tuples):
    # Read all configs
    train_filename = train_files[0].name
    configs_paths = [cur_p.parents[0] / '.hydra/config.yaml' for cur_p in train_files]
    configs_paths_bag = db.from_sequence(
        configs_paths, npartitions=(len(configs_paths)//100 + 1)  # 100 files per partition
    ).map(lambda p: (dict2df(read_yaml(p), prefix=None), p))  # Each entry in bag: (config_df, config_path)

    # Apply filters
    def filter_fn(config_path_tuple):
        config_df, config_path = config_path_tuple
        passed = True
        for cur_col, cur_filter in filter_tuples:
            passed = passed & np.all(cur_filter(config_df[cur_col]))
        return passed
    filtered_config_dfs_bag = configs_paths_bag.filter(filter_fn)

    # Get list of training files
    train_paths_bag = filtered_config_dfs_bag.map(lambda tup: tup[1].parents[1] / train_filename)
    return train_paths_bag


def load_run(run_path, config_prefix=None):
    """
    Given the path to a single df, load that df along with the .yaml file
    :param run_path: path to a single train.csv or eval.csv file
    :param config_prefix: None or string prefix to add to config column names
    :return: pandas dataframe of csv content with the associated config information
    """
    # Read the run (train or eval) as pandas df
    run_df = pd.read_csv(run_path)

    # Read the config file and process into a pandas df
    config_yaml_path = run_path.parents[0] / '.hydra/config.yaml'
    config_df = dict2df(read_yaml(config_yaml_path), prefix=config_prefix)

    # Join the two dfs
    cur_run_key = str(run_path.parents[0])  # making a common key based on the subdir they are both in
    run_df['subdir_key'] = cur_run_key  # NOTE: would this throw an error??
    config_df['subdir_key'] = cur_run_key
    run_config_df = run_df.merge(config_df, on='subdir_key', how='left')

    return run_config_df


def process_pd_df(pd_df):
    # Smooth episode reward (if present) by 100-long windows
    colname = 'episode_reward'
    rwindow = 100
    if colname in pd_df:
        smoothed_vals = pd_df[colname].rolling(rwindow).mean()
        smoothed_colname = f'smooth_{rwindow}_{colname}'
        pd_df[smoothed_colname] = smoothed_vals

    # Round steps to the nearest 1e4
    if False:
        colname = 'step'
        rstep = int(1e4)
        if colname in pd_df:
            round_vals = np.rint(pd_df[colname].values / rstep) * rstep
            round_colname = f'{colname}_round_{rstep}'
            pd_df[round_colname] = round_vals

    # Discard a proportion of the episodes
    if False:
        colname = 'episode'
        keep_every_k = 100
        if colname in pd_df:
            pd_df = pd_df[pd_df[colname] % keep_every_k == 0]

    # Convert the pandas df back into dictionary for easier dask concatenation later
    return pd_df.to_dict(orient='records')


def main():
    # Set up path and glob patterns
    base_dir = Path('/scratch/agc9824/project_outputs/url_benchmark/')
    glob_patterns = ['2022.09.19/133851_KVDO/*/*/train.csv',
                     '2022.09.20/153018_bzrn/*/*/train.csv']

    # Set up dask client
    client = Client()  # Client(threads_per_worker=4, n_workers=16)
    print(client)
    # Get all the run files
    print(f'File glob pattern: {base_dir}/{glob_patterns}')
    train_files = []
    for globpat in glob_patterns:
        train_files.extend(list(base_dir.glob(globpat)))
    train_files = sorted(train_files)
    # train_files = sorted(list(base_dir.glob(glob_pattern)))
    print(f'Number of files found: {len(train_files)}')

    # Get all the yaml config files
    configs_df = load_configs(train_files)
    describe_configs_df(configs_df)

    # Specify filters
    filter_tups = [
        ('agent.lr', lambda c: c == 1e-4),
        ('agent.critic_cfg.snd_kwargs.values_init', lambda c: c == 0.0),
    ]
    filtered_train_paths = get_filtered_runs_paths(train_files, filter_tups).compute()
    print('===== \nFiltered train paths:', len(filtered_train_paths))

    # Get the filtered training runs
    pd_df_bag = db.from_sequence(filtered_train_paths, npartitions=len(train_files)).map(load_run)
    pd_dict_bag = pd_df_bag.map(process_pd_df).flatten()
    print(f'pd_df_bag: {pd_df_bag}. pd_dict_bag: {pd_dict_bag}')

    # Construct pandas dataframe from bags
    da_df = pd_dict_bag.to_dataframe()
    pd_df = da_df.compute()
    pd_df = pd_df.reset_index(drop=True)
    print(pd_df)
    print(pd_df.columns)

    # ==
    # Plot
    print('=== Plotting ===')

    subplt_colname = 'task'
    subplt_list = sorted(list(pd_df[subplt_colname].unique()))
    plt.figure(figsize=(10, 7))
    for subplt_idx, subplt_name in enumerate(subplt_list):
        ax = plt.subplot(2, max(len(subplt_list)//2,1), subplt_idx + 1)
        cur_df = pd_df[pd_df[subplt_colname] == subplt_name]

        hue_str = 'agent.critic_cfg.snd_kwargs.temperature'
        sns.lineplot(x='step', y='critic_entropy_item_batch_avg',
                     hue=hue_str,
                     ci='sd',
                     palette='bright',
                     data=cur_df)

        ax.legend(title=hue_str.split('.')[-1], loc='upper right')

        plt.yscale('log')

        plt.title(subplt_name)

    plt.tight_layout()
    plt.savefig('tmp.png')


if __name__ == '__main__':
    main()