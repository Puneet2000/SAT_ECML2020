import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import numpy as np
# line 1 points 
# plt.figure(figsize=(5,5))
# x1 = [0.,0.2,0.4,0.6,0.8,1.] 
# y1 = [2.38,2.64,9.16,20.53,35.17,36.]

# f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# # plotting the line 1 points  
# x1 = np.linspace(0, 1.1, num=10, endpoint=False)
# plt.plot(x1, f(x1), label = "Resnet-10 std") 
  
# # line 2 points 
# x2 = [0.,0.2,0.4,0.6,0.8,1.] 
# y2 = [0.72,28.69,32.29,34.29,37.77,36.]

# f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# # plotting the line 1 points  
# x2 = np.linspace(0, 1.1, num=10, endpoint=False)
# plt.plot(x2, f(x2), label = "Resnet-10 adv") 

# x3 = [0.,0.2,0.4,0.6,0.8,1.] 
# y3 = [23.38,31.04,36.11,39.06,41.82,36.]

# f = interp1d(x3, y3,fill_value="extrapolate",kind='cubic')
# # plotting the line 1 points  
# x3 = np.linspace(0, 1.1, num=10, endpoint=False)
# plt.plot(x3, f(x3), label = "Resnet-10 sparse") 
  
# # naming the x axis 
# plt.ylabel('Robustness at $\epsilon$ = $\\frac{1}{255}$.') 
# # naming the y axis 
# plt.xlabel('$\\beta$') 

# plt.figure(figsize=(5,5))
# x1 = [0.,0.2,0.4,0.6,0.8,1.] 
# y1 = [0.27,0.43,2.26,7.52,18.0,16.07]

# f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# # plotting the line 1 points  
# x1 = np.linspace(0, 1.1, num=10, endpoint=False)
# plt.plot(x1, f(x1), label = "Resnet-10 std") 
  
# # line 2 points 
# x2 = [0.,0.2,0.4,0.6,0.8,1.] 
# y2 = [0.43,9.62,12.33,14.73,17.71,16.07]

# f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# # plotting the line 1 points  
# x2 = np.linspace(0, 1.1, num=10, endpoint=False)
# plt.plot(x2, f(x2), label = "Resnet-10 adv") 

# x3 = [0.,0.2,0.4,0.6,0.8,1.] 
# y3 = [9.07,14.56,18.26,22.46,24.48,16.07]

# f = interp1d(x3, y3,fill_value="extrapolate",kind='cubic')
# # plotting the line 1 points  
# x3 = np.linspace(0, 1.1, num=10, endpoint=False)
# plt.plot(x3, f(x3), label = "Resnet-10 sparse") 
  
# # naming the x axis 
# plt.ylabel('Robustness at $\epsilon$ = $\\frac{2}{255}$.') 
# # naming the y axis 
# plt.xlabel('$\\beta$') 


plt.figure(figsize=(5,5))
x1 = [0.,0.2,0.4,0.6,0.8,1.] 
y1 = [0.12,0.21,0.77,3.5,9.67,7.47]

f = interp1d(x1, y1,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x1 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x1, f(x1), label = "Resnet-10 std",linewidth=10) 
  
# line 2 points 
x2 = [0.,0.2,0.4,0.6,0.8,1.] 
y2 = [0.28,4.49,5.97,6.84,9.13,7.47]

f = interp1d(x2, y2,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x2 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x2, f(x2), label = "Resnet-10 adv",linewidth=10) 

x3 = [0.,0.2,0.4,0.6,0.8,1.] 
y3 = [4.54,7.72,9.7,13.29,13.54,7.47]

f = interp1d(x3, y3,fill_value="extrapolate",kind='cubic')
# plotting the line 1 points  
x3 = np.linspace(0, 1.1, num=10, endpoint=False)
plt.plot(x3, f(x3), label = "Resnet-10 sparse",linewidth=10) 
  
# naming the x axis 
plt.ylabel('Robustness at $\epsilon$ = $\\frac{3}{255}$.') 
# naming the y axis 
plt.xlabel('$\\beta$') 

# show a legend on the plot 
# plt.legend() 
  
# function to show the plot 
plt.savefig('alpha3.png') 
