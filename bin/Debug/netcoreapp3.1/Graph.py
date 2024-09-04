import matplotlib.pyplot as plt
logged_error = []
f = open("data.txt", "r")
for x in f:
  x = x.split(' ')
  set = []
  set.append(float(x[0]))
  set.append(float(x[1]))
  logged_error.append(set)

test = ([1,0.5],[2,0.25],[3,0.125])
fileName = "MLP"
x_data = []
y_data = []
x_data.extend([logged_error[i][0] for i in range(0,len(logged_error))])
y_data.extend([logged_error[i][1] for i in range(0,len(logged_error))])
fig, ax = plt.subplots()
fig.suptitle(fileName)
ax.set(xlabel='Epoch', ylabel='Squared Error')
ax.plot(x_data, y_data, 'tab:green')
plt.show()
