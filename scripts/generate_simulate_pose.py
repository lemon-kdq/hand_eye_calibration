import numpy as np
import os 
import argparse
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Coordinate Defination
# b: base  h: hand  w: world  e: eye

Rhe = Rotation.from_euler('zyx',np.array([90,3,2]),degrees=True).as_matrix()
the = np.array([0.5,1.25,0.23])
Rwb = Rotation.from_euler('zyx',np.array([45,10,20]),degrees=True).as_matrix()
twb = np.array([120.0,-102.0,23.3])

print(f'Rhe: {Rhe}')
print(f'the: {the}') 
# 生成时间序列 t
t = np.linspace(0, 10, 100)


#Tbh_i
t_bh_x = 10 * np.sin(t)
t_bh_y = 10 * np.cos(t)
t_bh_z = 1.0 * t

R_bh_yaw = 180 * np.sin(t)
R_bh_pitch = 2 * np.cos(t)
R_bh_roll = 4 * np.cos(t)

#Twe_i = Twb * Tbh_i * The




data_size = len(t)
with open('Tbh.txt','w') as bhf,open('Twe.txt','w') as wef:
    for i in range(0,data_size):
        q_bh_i = Rotation.from_euler('zyx',np.array([R_bh_yaw[i],R_bh_pitch[i],R_bh_roll[i]]),degrees=True).as_quat()
        R_bh_i = Rotation.from_quat(q_bh_i).as_matrix()
        t_bh_i = np.array([t_bh_x[i],t_bh_y[i],t_bh_z[i]])


        #Twh_i 
        R_wh_i = Rwb @ R_bh_i
        t_wh_i = Rwb @ t_bh_i + twb

        #Twe_i
        R_we_i = R_wh_i @ Rhe
        t_we_i = R_wh_i @ the + t_wh_i
        R_ew_i = R_we_i.T 
        t_ew_i = -1 * (R_ew_i @ t_we_i)
        q_ew_i = Rotation.from_matrix(R_ew_i).as_quat()
        

        T_bh_i = f'{t[i]} {t_bh_i[0]} {t_bh_i[1]} {t_bh_i[2]} {q_bh_i[0]} {q_bh_i[1]} {q_bh_i[2]} {q_bh_i[3]}\n'
        T_we_i = f'{t[i]} {t_ew_i[0]} {t_ew_i[1]} {t_ew_i[2]} {q_ew_i[0]} {q_ew_i[1]} {q_ew_i[2]} {q_ew_i[3]}\n'
        bhf.write(T_bh_i)
        wef.write(T_we_i)
    bhf.close()
    wef.close()
