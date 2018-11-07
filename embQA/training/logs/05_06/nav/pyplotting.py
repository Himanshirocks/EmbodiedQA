import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

iters = []
stat = []
stat1 = []

config = json.loads(open('11_03_06:00_pacman/train_1.json').read())

statistics = config['stats']
print(len(statistics))

for i in range(len(statistics)):
 iters.append(i)
 stat.append(statistics[i][0])
 stat1.append(statistics[i][1])
 
plt.figure(1)
plt.title('Training Progress (Navigation) in terms of Planner Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Loss')
plt.plot(iters,stat)
plt.savefig('planner_losses_underfit')

plt.figure(2)
plt.title('Training Progress (Navigation) in terms of Controller Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Loss')	
plt.plot(iters,stat1)
plt.savefig('controller_losses_underfit')

