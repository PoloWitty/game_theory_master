
from RL_brain import QLearningTable
from env import Env, TOTAL_SKILLS
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RL_brain import  simple_strategy
from parameter import INTERACTIVE

P_1=0.9
P_2=0.3


def update():
    for episode in tqdm(range(2000)):
        # initial observation
        t=0;state=0;betray=0
        observation=(t,state)#状态是(徒弟学会的技能数t,当前进行的轮数state,徒弟是否背叛)

        while state<TOTAL_SKILLS:

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            betray,apprentice_action,record,state_result,done=env.step(action,t,state,P_1,P_2)

            if apprentice_action:
                t+=1
            state+=1
            observation_=(t,state)
            if state==TOTAL_SKILLS-1:
                observation_='done'
            # RL learn from this transition
            RL.learn(str(observation), action, state_result[0], str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')

def sample():
    env.reset()
    #read the learning result
    RL.q_table=pd.read_csv('q_table.csv',index_col=0)
    ret_result=0
    # initial observation
    t = 0
    state = 0
    betray = 0
    observation = (t, state)  # 观测到的是(徒弟学会的技能数t,当前进行的轮数state,徒弟是否背叛)

    while state < TOTAL_SKILLS:

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # RL take action and get next observation and reward
        betray, apprentice_action, record, state_result, done = env.step(
            action, t, state, P_1, P_2)

        if apprentice_action:
            t += 1
        state += 1
        observation_ = (t, state)
        if state == TOTAL_SKILLS-1:
            observation_ = 'done'
        print('第',state,'轮','RL选择',action,'此轮收益为',state_result[0])
    
        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if done:
            ret_result=record.iloc[state-1,0]
            break
    return ret_result

def sample_compare():
    env.reset()
    state = 0
    t=0
    state_result=[0,0]
    ret_result=0
    while state<TOTAL_SKILLS:
        master_action=simple_strategy(state_result[0])

        apprentice_action=0
        betray=0
        betray,apprentice_action,record,state_result,done=env.step(master_action,t,state,P_1,P_2)

        if apprentice_action==1:
            t+=1
        state+=1

        # break while loop when end of this episode
        if done:
            ret_result=record.iloc[state-1,0]
            break
    return ret_result



if __name__ == "__main__":
    env = Env()
    RL = QLearningTable(actions=[1,0])
    update()
    RL.q_table.to_csv('q_table.csv')
    rl_res=sample()
    compare_res=sample_compare()
    print('强化学习结果总收益为',rl_res)
    print('简单策略结果总收益为',compare_res)
