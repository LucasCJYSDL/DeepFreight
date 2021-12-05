import numpy as np

def multi_step_TD(args, episode_buffer, round_num):

    n = args.step_num
    gamma = args.gamma
    for e in range(round_num):
        if (e + n) < round_num:
            episode_buffer['gamma'][e] = [gamma**n]
            temp_rwd = 0.
            for idx in range(e, e+ n):
                factor = gamma ** (idx - e)
                temp_rwd += factor * episode_buffer['r'][idx][0]
            episode_buffer['r'][e] = [temp_rwd]
            episode_buffer['next_idx'][e] = [n]
        else:
            episode_buffer['done'][e] = [True]
            episode_buffer['gamma'][e] = [gamma ** (round_num - e)]
            temp_rwd = 0.
            for idx in range(e, round_num):
                factor = gamma ** (idx - e)
                temp_rwd += factor * episode_buffer['r'][idx][0]
            episode_buffer['r'][e] = [temp_rwd]
            episode_buffer['next_idx'][e] = [round_num - 1 - e]  ## check
        if episode_buffer['next_idx'][e][0] + e - 1 < 0:
            print("Bad index!!!")
            episode_buffer['next_idx'][e][0] = 1 - e
    return episode_buffer
