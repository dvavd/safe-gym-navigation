import omnisafe
import numpy as np
from gym_navigation.envs.navigation_goal_safe import NavigationGoalSafe
import os
import json
import copy
from omnisafe.envs.wrapper import TimeLimit
from datetime import datetime
import sys # Required for stdout redirection
from contextlib import contextmanager # Required for context manager

@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close() # Important to close the file descriptor
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

    evaluator._env = TimeLimit(evaluator._env, time_limit)

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
        'hidden_sizes': [[64,64,32],[64,64]],
        'critic_lr': [0.0003, 0.0005],
        'update_iters': [10,20],
        'entropy_coef': [0.01, 0.05],
        'cost_limit': [1000.0, 5000.0],
        'batch_size': [64, 512],
        #'lagrangian_upper_bound': [1.0]
    }

    results_summary = []

    base_cfg = {
        "model_cfgs": {
            "actor_type": "gaussian_learning",
            "actor": {
                "lr": 0.0003,
                "hidden_sizes": [
                    64,
                    64
                ]
            },
            "critic": {
                "lr": 0.0003,
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
            "steps_per_epoch": 40000,
            "entropy_coef": 0.05,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": 10
        },
        "lagrange_cfgs": {
            'cost_limit': 1000.0,
            'lambda_lr': 0.01,
            'lagrangian_multiplier_init': 0.7
        },
    }

    for hidden_sizes_arr in param_grid['hidden_sizes']:
        for critic_lr_val in param_grid['critic_lr']:
            for update_iters_val in param_grid['update_iters']:
                for entropy_coef_val in param_grid['entropy_coef']:
                    for cost_limit_val in param_grid['cost_limit']:
                        for batch_size_val in param_grid['batch_size']:
                            current_params = {
                                'hidden_sizes': hidden_sizes_arr,
                                'critic_lr': critic_lr_val,
                                'update_iters': update_iters_val,
                                'entropy_coef': entropy_coef_val,
                                'cost_limit': cost_limit_val,
                                'batch_size': batch_size_val,
                            }
                            print(f"\nStarting run with params: {current_params}")

                            constrained_cfg = copy.deepcopy(base_cfg)
                            
                            constrained_cfg['model_cfgs']['actor']['hidden_sizes'] = hidden_sizes_arr
                            constrained_cfg['model_cfgs']['critic']['hidden_sizes'] = hidden_sizes_arr
                            constrained_cfg['model_cfgs']['critic']['lr'] = critic_lr_val
                            constrained_cfg['algo_cfgs']['entropy_coef'] = entropy_coef_val
                            constrained_cfg['lagrange_cfgs']['cost_limit'] = cost_limit_val
                            constrained_cfg['algo_cfgs']['batch_size'] = batch_size_val
                            
                            print("Starting constrained agent training...")
                            constrained_env_id = 'NavigationGoalSafe-v0'
                            constrained_agent = omnisafe.Agent('PPOLag', constrained_env_id, custom_cfgs=constrained_cfg)
                            constrained_agent.learn() 
                            print("Constrained agent training finished.")

                            eval_metrics = run_evaluation_and_save(
                                agent_log_dir=constrained_agent.agent._logger.log_dir,
                                num_eval_episodes=100,
                                eval_time_limit=1000,
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