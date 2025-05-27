import omnisafe
import numpy as np
from gym_navigation.envs.navigation_goal_safe import NavigationGoalSafe
import os
import json
import copy
from omnisafe.envs.wrapper import TimeLimit
from datetime import datetime
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def run_evaluation_and_save(
    agent_log_dir: str,
    num_episodes: int = 1000,
    time_limit: int = 1000,
):
    print(f"Starting evaluation for agent from: {agent_log_dir}")

    evaluator = omnisafe.Evaluator() 

    torch_save_dir = os.path.join(agent_log_dir, 'torch_save')

    if os.path.exists(torch_save_dir):
        pt_files = [f for f in os.listdir(torch_save_dir) if f.startswith('epoch-') and f.endswith('.pt')]
        if pt_files:
            pt_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]), reverse=True)
            model_name_to_load = pt_files[0]
        else:
            print(f"Error: No 'epoch-*.pt' files found in {torch_save_dir} either. Evaluation cannot proceed.")
            return None
    else:
        print(f"Error: torch_save directory not found at {torch_save_dir}. Evaluation cannot proceed.")
        return None

    evaluator.load_saved(
        save_dir=agent_log_dir,
        model_name=model_name_to_load,
    )

    evaluator._env = TimeLimit(evaluator._env, time_limit, device='cuda:0')

    with suppress_stdout():
        eval_results = evaluator.evaluate(num_episodes)

    episode_rewards, episode_costs, episode_lengths, episode_outcomes,  = eval_results[0], eval_results[1], eval_results[2], eval_results[3] 
    outcomes = {'timeout': 0, 'collision': 0, 'success': 0}
    outcome_counts = {k: episode_outcomes.count(k) for k in sorted(list(set(episode_outcomes)))}
    for outcome_type, count in outcome_counts.items():
        rate = count / num_episodes * 100
        outcomes[outcome_type] = rate

    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'avg_cost': np.mean(episode_costs),
        'avg_length': np.mean(episode_lengths),
        'collision_count': outcomes["collision"],
        'success_count': outcomes["success"],
        'timeout_count': outcomes["timeout"],
    }

    eval_file_path = os.path.join(agent_log_dir, "evaluation_results.json")
    try:
        with open(eval_file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Custom evaluation results saved to {eval_file_path}")
    except Exception as e:
        print(f"Error saving custom evaluation results: {e}")

    return metrics


def main():

    param_grid = {
        'kl_early_stop': [True, False],
    }

    results_summary = []

    base_cfg = {
        "model_cfgs": {
            "actor_type": "gaussian_learning",
            "linear_lr_decay": True,
            "exploration_noise_anneal": False,
            "actor": {
                "lr": 0.0006,
                "hidden_sizes": [
                    64,
                    64
                ]
            },
            "critic": {
                "lr": 0.00035,
                "hidden_sizes": [
                    64,
                    64
                ]
            }
        },
        "train_cfgs": {
            "total_steps": 2000000,
            "vector_env_nums": 1,
            "parallel": 1,
            "device": "cuda:0"
        },
        "algo_cfgs": {
            'batch_size': 512,
            "steps_per_epoch": 16384,
            "entropy_coef": 0.07,
            "max_grad_norm": 0.2,
            "reward_normalize": False,
            "cost_normalize": True,
            "obs_normalize": True,
            "kl_early_stop": False,
            "use_max_grad_norm": True,
            "use_critic_norm": True,
            "standardized_rew_adv": True,
            "standardized_cost_adv": True,
            "use_cost": True
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 10
        },
        "lagrange_cfgs": {
            'cost_limit': 0.09,
            'lambda_lr': 0.075,
            'lagrangian_multiplier_init': 0.1
        },
    }


    for kl_early_stop in param_grid['kl_early_stop']:
        current_params = {
            'kl_early_stop': kl_early_stop,
        }
        print(f"\nStarting run with params: {current_params}")

        constrained_cfg = copy.deepcopy(base_cfg)
    
        constrained_cfg['algo_cfgs']['kl_early_stop'] = kl_early_stop
        
        constrained_env_id = 'NavigationGoalSafe-v0'
        with suppress_stdout():
            constrained_agent = omnisafe.Agent('PPOLag', constrained_env_id, custom_cfgs=constrained_cfg)
            constrained_agent.learn() 
        print("Training finished.")

        eval_metrics = run_evaluation_and_save(
            agent_log_dir=constrained_agent.agent._logger.log_dir,
            num_episodes=1000,
            time_limit=1000,
        )

        if eval_metrics:
            run_summary = {
                'params': current_params,
                'metrics': eval_metrics,
                'log_dir': constrained_agent.agent._logger.log_dir
            }
            results_summary.append(run_summary)
            print(f"Finished run. Metrics: {eval_metrics}")
        else:
            print(f"Evaluation failed for params: {current_params}")

    # Save summary of all runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file_path = f"runs/grid_search_summary_{timestamp}.json"
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"\nGrid search summary saved to {summary_file_path}")
    except Exception as e:
        print(f"Error saving grid search summary: {e}")

if __name__ == '__main__':
    main()