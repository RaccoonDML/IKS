from importlib import import_module

from torch.utils.data.dataloader import default_collate
import torch

class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=False
            )


        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False, name=d)

            else:
                module_test = import_module('data.' + args.data_test.lower())
                testset = getattr(module_test, args.data_test)(args, train=False,name=d)

            self.loader_test.append(
                torch.utils.data.dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
