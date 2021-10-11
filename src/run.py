from trainer import Trainer
import sys

c = {
    'model_name': 'Resnet18','n_epoch': 70,
    'seed': [0], 'bs': [64], 'lr': [1e-5]
}

args = len(sys.argv)
if args >= 2:
    c['model_name'] = sys.argv[1]
    c['cv'] = int(sys.argv[2].split('=')[1])
    c['evaluate'] = int(sys.argv[3].split('=')[1])

trainer = Trainer(c)
trainer.run()