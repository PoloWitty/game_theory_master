import numpy as np
import pandas as pd
import time
import math
import sys
import matplotlib.pyplot as plt
from parameter import INTERACTIVE


G_APPRENTICE_BETRAY=10
G_MASTER_BETRAY=-10
TOTAL_SKILLS=20


class Env():
    def __init__(self):
        self.result=np.zeros(2)#第一个为师傅的累积收益, 第二个为徒弟的累积收益
        self.record=pd.DataFrame(np.zeros((TOTAL_SKILLS,2)))
        if INTERACTIVE:
            self.fig,self.ax=plt.subplots()
            plt.ion()

    def G_apprentice(self,t,apprentice_action,P_2):
        y=0
        # y=3
        y=-6*(1/(1+math.exp(-t+P_2*TOTAL_SKILLS))-0.5)+6/2#6为可变参数
        if apprentice_action==0:
            y=0
        return y
    
    def G_master(self,t,master_action,P_1):
        y=0
        if t<=P_1*TOTAL_SKILLS:
            y=8
        else:
            y=-t+P_1*TOTAL_SKILLS+8
        if master_action==0:
            y=0
        return y

    def _betray_possibility(self,t,P_2):
        y=0
        # if t<TOTAL_SKILLS:
        #     # y=0.8*(math.exp(t)-1)/(math.exp(TOTAL_SKILLS)-1)#0.8为可调超参
        #     y=0.5*t/TOTAL_SKILLS
        y=1/(1+math.exp(-t+P_2*TOTAL_SKILLS))
        return y
    
    def if_betray(self,t,P_2):
        p_=self._betray_possibility(t,P_2)
        # print('背叛概率为',p_)
        return np.random.choice([1,0],p=[p_,1-p_])
    
    def reset(self):
        self.result=np.zeros(2)
        self.record=pd.DataFrame(np.zeros_like(self.record))
        if INTERACTIVE:
            self.fig,self.ax=plt.subplots()
    
    def step(self,master_action,t,state,P_1,P_2):
        #计算徒弟的行为, 是学还是不学, 还是背叛
        apprentice_action_=1
        if master_action==1 :#教
            apprentice_action_=1#学
        elif master_action==0:#不教
            apprentice_action_=0#不学
        betray=self.if_betray(t,P_2)
        apprentice_action=apprentice_action_*(1-betray)

        done=0#游戏是否结束
        if betray==1 or state==TOTAL_SKILLS:
            done=1
        else:
            done=0
        
        #结算收益
        g_m=self.G_master(t,master_action,P_1)
        self.result[0]+=g_m
        if not betray:
            g_a=self.G_apprentice(t,apprentice_action,P_2)
            self.result[1]+=g_a

        if betray:
            g_m=G_MASTER_BETRAY
            self.result[0]+=g_m
            g_a=G_APPRENTICE_BETRAY
            self.result[1]+=g_a
        
        state_result=[]
        state_result.append(g_m)    
        state_result.append(g_a)    

        self.record.iloc[state,0]=self.result[0]
        self.record.iloc[state,1]=self.result[1]

        return betray,apprentice_action,self.record,state_result,done
        
    def render(self):
        if INTERACTIVE:
            self.ax.cla()
            self.record.plot(kind='bar',ax=self.ax)
            plt.pause(0.2)
        else:
            pass
