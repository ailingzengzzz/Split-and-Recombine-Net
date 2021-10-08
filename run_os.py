from subprocess import call
import sys
action = ['Greeting','Sitting','SittingDown','WalkTogether','Phoning','Posing','WalkDog','Walking','Purchases','Waiting','Directions','Smoking','Photo','Eating','Discussion']
group = [1, 2, 3, 5]

for i in action:
     cmd = """nohup python -u run.py --model srnet -arc 1,1,1 --use-action-split True --train-action {} -mn sr_t1_crossaction_act{} > log/sr_t1_crossaction_act{}.log 2>&1&""".format(i,i,i)
     print(cmd)
     call(cmd, shell=True)
print('Finish!')

