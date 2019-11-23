import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import numpy as np
 
plt.figure(figsize=(5,5))
x1 = [0.,0.2,0.4,0.6,0.8,1.] 
y1 = [2.38,2.64,9.16,20.53,35.17,33.15]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linestyle='--',color='red',linewidth=5)
  
# line 2 points 
x2 = [0.,0.2,0.4,0.6,0.8,1.] 
y2 = [0.72,28.69,32.29,34.29,37.77,33.15]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linestyle='-',color='red',linewidth=5)  




x1 = [0.,0.2,0.4,0.6,0.8,1.] 
y1 = [0.27,0.43,2.26,7.52,18.0,13.70]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linestyle='--',color='blue',linewidth=5)
  
# line 2 points 
x2 = [0.,0.2,0.4,0.6,0.8,1.] 
y2 = [0.43,9.62,12.33,14.73,17.71,13.70]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linestyle='-',color='blue',linewidth=5) 




x1 = [0.,0.2,0.4,0.6,0.8,1.] 
y1 = [0.12,0.21,0.77,3.5,9.67,6.01]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linestyle='--',color='green',linewidth=5) 
  
# line 2 points 
x2 = [0.,0.2,0.4,0.6,0.8,1.] 
y2 = [0.28,4.49,5.97,6.84,9.13,6.01]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linestyle='-',color='green',linewidth=5)  
 
  
# naming the x axis 
plt.ylabel('Robustness') 
# naming the y axis 
plt.xlabel('$\\alpha^{10}$') 

# show a legend on the plot 
# plt.legend() 
  
# function to show the plot 
plt.savefig('graph.png') 
