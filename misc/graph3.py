import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import numpy as np
 
plt.figure(figsize=(5,5))
x1 = [0,2,4,8,16] 
y1 = [25.83	, 30.21, 30.63, 20.53 ,4.51]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 17, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linestyle='--',color='red',linewidth=5)
  
# line 2 points 
x2 = [0,2,4,8,16] 
y2 = [25.83, 28.11, 30.7, 34.29, 13.75]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 17, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linestyle='-',color='red',linewidth=5)  




x1 = [0,2,4,8,16] 
y1 = [7.76,12.59,11.31,7.52,1.08]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 17, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linestyle='--',color='blue',linewidth=5)
  
# line 2 points 
x2 = [0,2,4,8,16] 
y2 = [7.76,9.3,10.45,14.73,2.86]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 17, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linestyle='-',color='blue',linewidth=5) 




x1 = [0,2,4,8,16] 
y1 = [3.35,6.69,5.26,3.5,0.44]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 17, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linestyle='--',color='green',linewidth=5) 
  
# line 2 points 
x2 = [0,2,4,8,16] 
y2 = [3.35,3.85,4.42,6.84,1.01]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 17, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linestyle='-',color='green',linewidth=5)  
 
  
# naming the x axis 
plt.ylabel('Robustness',fontsize=16, fontweight='bold') 
# naming the y axis 
plt.xlabel('$\\epsilon_0$',fontsize=16, fontweight='bold') 


# plt.legend() 
  
# function to show the plot 
plt.savefig('graph2.png') 
