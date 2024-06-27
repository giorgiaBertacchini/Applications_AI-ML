import matplotlib.pyplot as plt


def plt_reward_over_episodes(rewards: list[float], title: str):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title(title)
    plt.show()
