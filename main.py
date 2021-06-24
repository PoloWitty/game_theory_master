from env import Env, TOTAL_SKILLS
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RL_brain import  simple_strategy
from parameter import INTERACTIVE

P_2_TEST=10
TOTAL_EPISODE=10
BATCH_SIZE=10

def update():
    log=pd.DataFrame(np.zeros((P_2_TEST*TOTAL_EPISODE,3)),columns=['p_2','p_1',"师傅的总收益"])
    # if not INTERACTIVE:
    P_1=0
    P_2=0
    for i in range(P_2_TEST):#测试p_2
        P_1=0
        for episode in range(TOTAL_EPISODE):
            if INTERACTIVE:
                print('episode: ',episode)
                print('P_1: ',P_1)
            # initial observation
            sum_gain=0
            for batch in range(BATCH_SIZE):
                if INTERACTIVE:
                    env.render()
                state=0
                t=0
                state_result=[0,0]
                record = pd.DataFrame(np.zeros((TOTAL_SKILLS, 2)))
                while state<TOTAL_SKILLS:

                    if INTERACTIVE:
                        master_action=int(input('此轮你的决策是:'))#人机交互版本
                    else:
                        master_action=simple_strategy(state_result[0])

                    apprentice_action=0
                    betray=0
                    betray,apprentice_action,record,state_result,done=env.step(master_action,t,state,P_1,P_2)

                    if apprentice_action==1:
                        t+=1
                    if INTERACTIVE:
                        if betray==1:
                            print('徒弟此轮背叛了你')
                        else:
                            print("徒弟此轮选择了",apprentice_action)
                        print('此轮你的收益为', state_result[0], '徒弟的收益为', state_result[1])
                    state+=1
                    if INTERACTIVE:
                        print('徒弟学到的技能数为:',t,' 当前轮数:',state)
                        print('\n')

                    if INTERACTIVE:
                        env.render()


                    # break while loop when end of this episode
                    if done:
                        break

                # end of game
                # print('game over')
                sum_gain+=record.iloc[state-1,0]
                env.reset()
            log.iloc[i*P_2_TEST+episode, 0]=P_2
            log.iloc[i*P_2_TEST+episode, 1] = P_1
            log.iloc[i*P_2_TEST+episode, 2] = sum_gain/BATCH_SIZE
            P_1+=1/TOTAL_EPISODE
        P_2+=1/P_2_TEST
    log.to_csv('log.csv')


if __name__ == "__main__":
    env = Env()
    update()
