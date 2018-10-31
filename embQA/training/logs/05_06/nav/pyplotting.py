import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

iters = []
stat = []

config = json.loads(open('10_30_04:12_pacman/train_1.json').read())


statistics = config['stats']


plt.title('Training Progress (Navigation) in terms of Planner Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Loss')	
		


for i in range(len(statistics)):
 print(i)
 iters.append(i)
 print(statistics[i][0])
 stat.append(statistics[i][0])
 

plt.plot(iters,stat)
plt.savefig('planner_losses')


iters1 = []
stat1 = []
plt.title('Training Progress (Navigation) in terms of Controller Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Loss')	
		


for i in range(len(statistics)):
 print(i)
 iters1.append(i)
 print(statistics[i][1])
 stat1.append(statistics[i][1])
 

plt.plot(iters1,stat1)
plt.savefig('controller_losses')

