import optuna
import torch
import numpy as np
import omnisafe
from gym_navigation.envs.navigation_goal_safe import NavigationGoalSafe
import time
from contextlib import contextmanager
import os
import sys
import rich.progress

original_progress = rich.progress.Progress

def disabled_progress(*args, **kwargs):
    kwargs["disable"] = True
    return original_progress(*args, **kwargs)

rich.progress.Progress = disabled_progress

class OmniSafeOptunaCallback:
    def __init__(self, trial, eval_freq=1):
        self.trial = trial
        self.eval_freq = eval_freq
        self.is_pruned = False

    def __call__(self, agent, epoch):
        # report every eval_freq epochs
        if epoch % self.eval_freq == 0:
            ep_reward = agent.agent._logger.get_stats("Metrics/EpRet")[0]
            ep_cost = agent.agent._logger.get_stats("Metrics/EpCost")[0]
            print(f"Epoch: {epoch}, EpRet: {ep_reward}, EpCost: {ep_cost}")
            metric_value =  ep_reward - 200 * ep_cost
            
            self.trial.report(metric_value, epoch)
            
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return False


def objective(trial):
    net_arch = {
        "one_layer": [32],
        "one_layer2": [64],
        "two_layer": [64,32],
        "two_layer2": [64, 64],
        "three_layer": [64, 32, 16],
        "three_layer2": [64, 64, 32],
    }
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-3)
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-3)
    lr_lambda = trial.suggest_categorical("lagrangian_multiplier_init", [0.01, 0.1, 0.5, 1.0])
    hidden_sizes_actor = trial.suggest_categorical("hidden_size", net_arch.keys())
    hidden_sizes_critic = trial.suggest_categorical("hidden_size", net_arch.keys())
    lagrangian_multiplier_init = trial.suggest_float("lr_lambda", 1e-3, 1e-1) 
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    steps_per_epoch = trial.suggest_categorical("steps_per_epoch", [1024, 2048, 4096, 8192, 16384])
    entropy_coef = trial.suggest_float("entropy_coef", 0.01, 0.1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.01, 1.0)
    cost_limit = trial.suggest_float("cost_limit", 0.0, 1.0)
    reward_normalize = trial.suggest_categorical("reward_normalize", [True, False])
    cost_normalize = trial.suggest_categorical("cost_normalize", [True, False])
    obs_normalize = trial.suggest_categorical("obs_normalize", [True, False])
    kl_early_stop = trial.suggest_categorical("kl_early_stop", [True, False])
    use_max_grad_norm = trial.suggest_categorical("use_max_grad_norm", [True, False])
    use_critic_norm = trial.suggest_categorical("use_critic_norm", [True, False])
    standardized_rew_adv = trial.suggest_categorical("standardized_rew_adv", [True, False])
    standardized_cost_adv = trial.suggest_categorical("standardized_cost_adv", [True, False])
    use_cost = trial.suggest_categorical("use_cost", [True, False])
    linear_lr_decay = trial.suggest_categorical("linear_lr_decay", [True, False])
    exploration_noise_anneal = trial.suggest_categorical("exploration_noise_anneal", [True, False])

    custom_cfgs = {
        "model_cfgs": {
            "actor_type": "gaussian_learning",
            "linear_lr_decay": linear_lr_decay,
            "exploration_noise_anneal": exploration_noise_anneal,
            "actor": {
                "lr": lr_actor,
                "hidden_sizes": net_arch[hidden_sizes_actor],
            },
            "critic": {
                "lr": lr_critic,
                "hidden_sizes": net_arch[hidden_sizes_critic],
            }
        },
        "train_cfgs": {
            "total_steps": 1000000,
            "vector_env_nums": 1,
            "parallel": 1,
            "device": "cpu"
        },
        "algo_cfgs": {
            'batch_size': batch_size,
            "steps_per_epoch": steps_per_epoch,
            "entropy_coef": entropy_coef,
            "max_grad_norm": max_grad_norm,
            "reward_normalize": reward_normalize,
            "cost_normalize": cost_normalize,
            "obs_normalize": obs_normalize,
            "kl_early_stop": kl_early_stop,
            "use_max_grad_norm": use_max_grad_norm,
            "use_critic_norm": use_critic_norm,
            "standardized_rew_adv": standardized_rew_adv,
            "standardized_cost_adv": standardized_cost_adv,
            "use_cost": use_cost
        },
        "lagrange_cfgs": {
            "cost_limit": cost_limit,
            "lagrangian_multiplier_init": lagrangian_multiplier_init,
            "lambda_lr": lr_lambda,
            "lambda_optimizer": "Adam",
            "lagrangian_upper_bound": None
        },
    }
            
    agent = omnisafe.Agent('PPOLag', 'NavigationGoalSafe-v0', custom_cfgs=custom_cfgs)
    
    callback = OmniSafeOptunaCallback(trial, eval_freq=1)
    
    # monkey patch the agent's learn method to check for pruning
    original_learn = agent.learn
    
    def learn_with_pruning():
        start_time = time.time()
        agent.agent._logger.log('INFO: Start training')
        try:
            for epoch in range(agent.agent._cfgs.train_cfgs.epochs):
                epoch_time = time.time()
                rollout_time = time.time()

                agent.agent._env.rollout(
                    steps_per_epoch=agent.agent._steps_per_epoch,
                    agent=agent.agent._actor_critic,
                    buffer=agent.agent._buf,
                    logger=agent.agent._logger,
                )
                agent.agent._logger.store({'Time/Rollout': time.time() - rollout_time})
                update_time = time.time()
                agent.agent._update()
                agent.agent._logger.store({'Time/Update': time.time() - update_time})

                if agent.agent._cfgs.model_cfgs.exploration_noise_anneal:
                    agent.agent._actor_critic.annealing(epoch)

                if agent.agent._cfgs.model_cfgs.actor.lr is not None:
                    agent.agent._actor_critic.actor_scheduler.step()
                
                if callback(agent, epoch):
                    raise optuna.exceptions.TrialPruned()
                
                agent.agent._logger.store(
                    {
                        'TotalEnvSteps': (epoch + 1) * agent.agent._cfgs.algo_cfgs.steps_per_epoch,
                        'Time/FPS': agent.agent._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                        'Time/Total': (time.time() - start_time),
                        'Time/Epoch': (time.time() - epoch_time),
                        'Train/Epoch': epoch,
                        'Train/LR': (
                            0.0
                            if agent.agent._cfgs.model_cfgs.actor.lr is None
                            else agent.agent._actor_critic.actor_scheduler.get_last_lr()[0]
                        ),
                    },
                )

                # not needed 
                #agent.agent._logger.dump_tabular()
            
            ep_ret = agent.agent._logger.get_stats('Metrics/EpRet')[0]
            ep_cost = agent.agent._logger.get_stats('Metrics/EpCost')[0]
            ep_len = agent.agent._logger.get_stats('Metrics/EpLen')[0]
            agent.agent._logger.close()
            agent.agent._env.close()

            return ep_ret, ep_cost, ep_len
        
        finally:
            try:
                agent.agent._logger.close()
            except Exception:
                pass
            try:
                agent.agent._env.close()
            except Exception:
                pass
    
    # replace learn method
    agent.agent.learn = learn_with_pruning
    
    try:
        ep_ret, ep_cost, _ = agent.learn()
        return ep_ret - 200 * ep_cost # make sure the value is right
    except optuna.exceptions.TrialPruned:
        raise
    except AssertionError as e:
        # handle the "cost for updating lagrange multiplier is nan" error
        if "cost for updating lagrange multiplier is nan" in str(e):
            print(f"Trial failed with AssertionError: {e}. Returning negative infinity score to continue optimization.")
            return float('-inf') 
        else:
            raise

storage = "sqlite:///safe_goal_navigation_study.db"
study_name = "safe_goal_navigation"
study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=10, n_jobs=4)