from utils import TrainValDataLoader
from utils import GetArgs, check_dir, seq_collate_typeA
import torch
from torch.utils.data import DataLoader
from trainer import Trainer


class Fit:
    def __init__(self):
        self.args = GetArgs().getargs()

    def train(self):
        folder_name = f'weight/' + self.args.dataset_type + '_' + self.args.model_name + '_model' + str(
            self.args.exp_id)
        check_dir(folder_name)

        train_loader = TrainValDataLoader(args=self.args)
        self.args.num_train_scenes = train_loader.num_scene

        val_loader = TrainValDataLoader(args=self.args, validation=True)
        self.args.num_val_scenes = val_loader.num_scene

        train_iter = DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_cores, drop_last=False, collate_fn=seq_collate_typeA)
        val_iter = DataLoader(val_loader, batch_size=self.args.batch_size*4, shuffle=False,
                              num_workers=self.args.num_cores, drop_last=False, collate_fn=seq_collate_typeA)

        trainer = Trainer(self.args)
        trainer.train(train_iter, val_iter)

        pass


if __name__ == '__main__':
    Fit().train()

    pass
