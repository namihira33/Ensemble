from trainer import Trainer
import sys

c = {
    'model_name': ['Resnet18'],'n_epoch': list(range(10,11)),
    'seed': [0], 'bs': 64, 'lr': [1e-4]
}

args = len(sys.argv)
if args >= 2:
    c['model_name'] = sys.argv[1]

trainer = Trainer(c)
trainer.run()