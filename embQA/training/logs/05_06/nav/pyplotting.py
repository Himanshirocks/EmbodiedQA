import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

iters = []
stat = []

config = json.loads(open('10_30_04:12_pacman/train_1.json').read())


statistics = config['stats']


plt.title('Training Progress in terms of Planner Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Loss')	
		


for i in range(len(statistics)):
 i
 iters.append(i)
 statistics[i][0]
 stat.append(statistics[i][0])
 

plt.plot(iters,stat)
plt.savefig('myfig')

