from pdb import set_trace as TT
import numpy as np
from ray.rllib.algorithms.ppo import PPO as RlLibPPOTrainer
import torchinfo
import torch as th

# class TrialProgressReporter(CLIReporter):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_timesteps = []

#     def should_report(self, trials, done=False):
#         old_num_timesteps = self.num_timesteps
#         self.num_timesteps = [t.last_result['timesteps_total'] if 'timesteps_total' in t.last_result else 0 for t in trials]
#         # self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
#         done = np.any(self.num_timesteps > old_num_timesteps)
#         return done


class PPOTrainer(RlLibPPOTrainer):
    log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # wandb.init(**self.config['wandb'])
        self.checkpoint_path_file = kwargs['config']['checkpoint_path_file']
        self.ctrl_metrics = self.config['env_config']['controls']
        self.ctrl_metrics = {} if self.ctrl_metrics is None else self.ctrl_metrics
        cbs = self.workers.foreach_env(lambda env: env.unwrapped.cond_bounds)
        cbs = [cb for worker_cbs in cbs for cb in worker_cbs if cb is not None]
        cond_bounds = cbs[0]
        self.metric_ranges = {k: v[1] - v[0] for k, v in cond_bounds.items()}
        # self.checkpoint_path_file = checkpoint_path_file

    def setup(self, config):
        ret = super().setup(config)
        n_params = 0
        param_dict = self.get_weights()['default_policy']

        for v in param_dict.values():
            n_params += np.prod(v.shape)
        model = self.get_policy('default_policy').model
        print(f'default_policy has {n_params} parameters.')
        print('Model overview(s):')
        print(model)
        print("=============")
        # torchinfo summaries are very confusing at the moment
        torchinfo.summary(model, input_data={
            "input_dict": {"obs": th.zeros((1, *self.config['model']['custom_model_config']['dummy_env_obs_space'].shape))}})
        return ret

    @classmethod
    def get_default_config(cls):
        # def_cfg = super().get_default_config()
        def_cfg = RlLibPPOTrainer.get_default_config()
        def_cfg.update({
            'checkpoint_path_file': None,
            'wandb': {
                'project': 'PCGRL',
                'name': 'default_name',
                'id': 'default_id',
            },
        })
        return def_cfg

    def save(self, *args, **kwargs):
        ckp_path = super().save(*args, **kwargs)
        with open(self.checkpoint_path_file, 'w') as f:
            f.write(ckp_path)
        return ckp_path

    # @wandb_mixin
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        log_result = {k: v for k, v in result.items() if k in self.log_keys}
        log_result['info: learner:'] = result['info']['learner']

        # Either doing multi-agent...
        if 'num_agent_steps_sampled_this_iter' in result:
            result['fps'] = result['num_agent_steps_trained_this_iter'] / result['time_this_iter_s']
        # or single-agent.
        else:
            result['fps'] = result['num_env_steps_trained_this_iter'] / result['time_this_iter_s']

        # TODO: Send a heatmap to tb/wandb representing success reaching various control targets?
        if len(result['custom_metrics']) > 0:
            n_bins = 20
            result['custom_plots'] = {}
            for metric in self.ctrl_metrics:

                # Scatter plots via wandb
                # trgs = result['hist_stats'][f'{metric}-trg']
                # vals = result['hist_stats'][f'{metric}-val']
                # data = [[x, y] for (x, y) in zip(trgs, vals)]
                # table = wandb.Table(data=data, columns=['trg', 'val'])
                # scatter = wandb.plot.scatter(table, "trg", "val", title=f"{metric}-trg-val")
                # result['custom_plots']["scatter_{}".format(metric)] = scatter
                # scatter.save(f"{metric}-trg-val.png")
                # wandb.log({f'{metric}-scc': scatter}, step=self.iteration)

                # Spoofed histograms
                # FIXME: weird interpolation behavior here???
                bin_size = self.metric_ranges[metric] / n_bins  # 30 is the default number of tensorboard histogram bins (HACK)
                trg_dict = {}

                for i, trg in enumerate(result['hist_stats'][f'{metric}-trg']):
                    val = result['hist_stats'][f'{metric}-val'][i]
                    scc = 1 - abs(val - trg) / self.metric_ranges[metric]
                    trg_bin = trg // bin_size
                    if trg not in trg_dict:
                        trg_dict[trg_bin] = [scc]
                    else:
                        trg_dict[trg_bin] += [scc]
                # Get average success rate in meeting each target.
                trg_dict = {k: np.mean(v) for k, v in trg_dict.items()}
                # Repeat each target based on how successful we were in reaching it. (Appears at least once if sampled)
                spoof_data = [[trg * bin_size] * (1 + int(20 * scc)) for trg, scc in trg_dict.items()]
                spoof_data = [e for ee in spoof_data for e in ee]  # flatten the list
                result['hist_stats'][f'{metric}-scc'] = spoof_data

                # Make a heatmap.
                # ax, fig = plt.subplots(figsize=(10, 10))
                # data = np.zeros(n_bins)
                # for trg, scc in trg_dict.items():
                    # data[trg] = scc
                # wandb.log({f'{metric}-scc': wandb.Histogram(data, n_bins=n_bins)})

                # plt.imshow(data, cmap='hot')
                # plt.savefig(f'{metric}.png')

            

        # for k, v in result['hist_stats'].items():
            # if '-trg' in k or '-val' in k:
                # result['custom_metrics'][k] = [v]

        # print('-----------------------------------------')
        # print(pretty_print(log_result))
        return result

    def evaluate(self):
        # TODO: Set the evaluation maps here!
        # self.eval_workers.foreach_env_with_context(fn)
        result = super().evaluate()
        return result
