import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt


def smooth(scalars, weight=0.971):
    """
    Reference: https://github.com/plotly/dash-live-model-training/blob/master/app.py#L163
    """
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def get_section_results(file, title):
    """
        requires tensorflow==1.12.0
    """
    logdir = os.path.join("data", file, "events*")
    file = glob.glob(logdir)[-1]
    returns = []

    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == title:
                returns.append(v.simple_value)
    return returns


dir = "/home/ayush/Documents/bipedal-hrl/HACKerMan/Hierarchical-Actor-Critic-HAC-PyTorch/logs/"

# NORMAL
reward_norm_async = get_section_results(
    dir + "MountainCarContinuous-h-v1_normal_async_20211215-182955", "Eval_Reward"
)
steps_norm_async = get_section_results(
    dir + "MountainCarContinuous-h-v1_normal_async_20211215-182955", "Eval_Num_Steps",
)
reward_norm_sync = get_section_results(
    dir + "MountainCarContinuous-h-v1_normal_sync_20211215-183009", "Eval_Reward"
)
steps_norm_sync = get_section_results(
    dir + "MountainCarContinuous-h-v1_normal_sync_20211215-183009", "Eval_Num_Steps",
)


# OU
reward_norm_async = get_section_results(
    dir + "MountainCarContinuous-h-v1_ou_async_20211215-183029", "Eval_Reward"
)
steps_norm_async = get_section_results(
    dir + "MountainCarContinuous-h-v1_ou_async_20211215-183029", "Eval_Num_Steps",
)
reward_norm_sync = get_section_results(
    dir + "MountainCarContinuous-h-v1_ou_sync_20211215-183049", "Eval_Reward"
)
steps_norm_sync = get_section_results(
    dir + "MountainCarContinuous-h-v1_ou_sync_20211215-183049", "Eval_Num_Steps",
)

# SURPRISE
reward_norm_async = get_section_results(
    dir + "MountainCarContinuous-h-v1_surprise_async_20211215-184217", "Eval_Reward"
)
steps_norm_async = get_section_results(
    dir + "MountainCarContinuous-h-v1_surprise_async_20211215-184217", "Eval_Num_Steps",
)
reward_norm_sync = get_section_results(
    dir + "MountainCarContinuous-h-v1_surprise_async_20211215-184217", "Eval_Reward"
)
steps_norm_sync = get_section_results(
    dir + "MountainCarContinuous-h-v1_surprise_async_20211215-184217", "Eval_Num_Steps",
)

plt.plot(reward_norm_async, label="Asynchronous Gaussian Exploration")
plt.plot(reward_norm_sync, label="Synchronous Gaussian Exploration")
plt.title("Eval Reward vs. Steps")
plt.xlabel("Num Steps")
plt.ylabel("Eval Reward")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig("norm_reward.png", bbox_inches="tight")
