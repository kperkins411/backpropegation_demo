import numpy as np
import matplotlib.pyplot as plt

# a=2,b=3,w1=.01,w2=.02, target= 79
#going to train the following single node net to
#output the correct value, keep in mind a,b and target
#are fixed, can only change w1 and w2
# f= a*w1*b*w2

# error function
# error = 1/2(target-f)*(target-f)

# derivitives
# d(error)/df = (target-f)*(-1)
# d(f)/dw1 = a*b*w2
# d(f)/dw2 = a*b*w1

#initial params
learning_rate     = .001    #set this too high and the error explodes
number_iterations = 1000
stop_when_error_less_than = .0001  #will converge since only 1 set of numbers

a=2.0
b=3.0
target= 79.0

#helper functions
#f = lambda a,b,w1,w2: a*w1*b*w2                     #f= a*w1*b*w2
f_error = lambda target,f: 1/2*(target-f)*(target-f)   # error = 1/2(target-f)*(target-f)

#stuff that changes
w1=.1
w2=.2
list_errors = []

for i in range(number_iterations):
    #forward
    f = a*w1*b*w2
    e = f_error(target,f)   #1/2*(target-f)*(target-f)
    if (e<stop_when_error_less_than):
        break

    #save for plotting
    list_errors.append(e)
    print ("error is " + str(e))

    #backward
    derr = (target-f)*(-1)
    w1 += learning_rate*-1.0*a*b*w2*derr
    w2 += learning_rate*-1.0*a*b*w1*derr
# y=range(len(list_errors))
# plt.scatter(list_errors, y)
#x,y = zip(*list_errors)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("backpropegate with1 neuron")
ax.set_xlabel("Iterations")
ax.set_ylabel("Error")
ax.text(0.9, 0.6, "final w1={:.4}".format(w1) + "\nfinal w2={:.4}".format(w2) + "\na="+ str(a) + "\nb=" + str(b)+ "\ntarget=" + str(target),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='yellow', fontsize=15,style='italic',bbox={'facecolor':'black', 'alpha':0.5, 'pad':10})

ax.plot(range(len(list_errors)), list_errors)
plt.show()