import copy
import os
import random
import sys
import time
import numpy as np

import torch
import wandb
from torch.utils.data import DataLoader

import model
import dataset
from metrics.question_answer import question_answer
from model.loss import calc_qa_loss
from utils.container import metricsContainer
from utils.helper import move_to_cuda
from utils.timer import Timer


class MainRunner:
    def __init__(self, config, use_wandb):
        print("Initialization Start.")
        self.config = config
        self.use_wandb = use_wandb
        self._init_misc()
        self._init_dataset(**config.dataset)
        self._init_model(config.model)
        self._init_optimizer(config.optimizer)
        print("Initialization End.")

    def _init_dataset(self, shared_cfg, train_data_cfg, val_data_cfg, test_data_cfg, **kwargs):
        # Done √
        val_data_cfg.update(shared_cfg)
        test_data_cfg.update(shared_cfg)
        assert isinstance(train_data_cfg, list), "Incomplete data config!"
        self.train_dataset = []
        self.train_loader = []
        for bucket_cfg in train_data_cfg:
            bucket_cfg.update(shared_cfg)
            self.train_dataset.append(getattr(dataset, bucket_cfg.name, None)(**bucket_cfg, action="train"))
        self.val_dataset = getattr(dataset, val_data_cfg.name, None)(**val_data_cfg, action="val")
        self.test_dataset = getattr(dataset, test_data_cfg.name, None)(**test_data_cfg, action="test")
        print("Train: {} samples; Val: {} samples, Test: {} samples".format(sum([len(train_data) for train_data in self.train_dataset]),
                                                                            len(self.val_dataset),
                                                                            len(self.test_dataset)))
        for train_dataset in self.train_dataset:
            self.train_loader.append(DataLoader(train_dataset, batch_size=self.train_config.batch_size, shuffle=True,
                                                collate_fn=train_dataset.collate_data, num_workers=1))
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.train_config.batch_size, shuffle=False,
                                     collate_fn=self.val_dataset.collate_data, num_workers=1)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_config.batch_size, shuffle=False,
                                      collate_fn=self.test_dataset.collate_data, num_workers=1)

    def _init_model(self, model_config):
        self.model = getattr(model, model_config.name, None)(**model_config)
        self.model = self.model.cuda(device=0)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        if self.use_wandb:
            wandb.watch(self.model)
        print(self.model)

    def _export_log(self, epoch, total_step, batch_idx, lr, loss_meter, acc, time_meter):
        msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, batch_idx, lr)
        for k, v in loss_meter.items():
            msg += '{} = {:.4f}, '.format(k, v)
        msg += 'acc = {:.4f}, '.format(acc)
        msg += '{:.3f} seconds/batch'.format(time_meter)
        print(msg)
        sys.stdout.flush()
        loss_meter.update({"batch": total_step})
        if self.use_wandb:
            wandb.log(loss_meter)

    def _print_metrics(self, epoch, metrics):
        msg = "Epoch {}".format(epoch)
        for k, v in metrics.items():
            msg += ', {} = {:.4f}'.format(k, v)
        print(msg)
        sys.stdout.flush()
        metrics.update({"epoch": epoch})
        if self.use_wandb:
            wandb.log(metrics)

    def _init_optimizer(self, optimizer_config):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule

        self.optimizer = AdamOptimizer(optimizer_config, list(self.model.parameters()))
        self.lr_scheduler = InverseSquareRootSchedule(optimizer_config, self.optimizer)
        self.loss_config = optimizer_config.loss_config

    def _init_misc(self):
        seed = 8
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(self.config)
        if self.use_wandb:
            wandb.init(project="CVPR2021")
            wandb.config.update(self.config)
        self.train_config = self.config.train
        self.test_config = self.config.test
        self.model_saved_path = self.train_config["saved_path"]
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        print('GPU: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        self.num_updates = 0

    def _train_one_epoch(self, epoch, last_total_step):
        self.model.train()
        timer = Timer()
        total_batch_idx = 0
        for train_loader in self.train_loader:
            for _, batch in enumerate(train_loader, 1):
                total_batch_idx += 1
                self.optimizer.zero_grad()
                net_input = move_to_cuda(batch['net_input'])
                output = self.model(**net_input)
                loss, loss_items = calc_qa_loss(avg_pred=output["avg_result"],
                                                all_pred=output["all_result"],
                                                target=batch["target"]["answer"])
                predictions = output["avg_result"].squeeze().argmax(dim=-1).long()
                correct = torch.sum(predictions.eq(batch["target"]["answer"].to(predictions.device))).float()
                acc = (correct * 100.0 / batch["target"]["answer"].size(0)).item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                # update
                self.optimizer.step()
                self.num_updates += 1
                curr_lr = self.lr_scheduler.step_update(self.num_updates)
                time_interval = timer.elapsed_interval
                timer.reset()
                metricsContainer.update("loss", loss_items)
                metricsContainer.update("acc", acc)
                metricsContainer.update("train_time", time_interval)

                if total_batch_idx % self.train_config.display_interval == 0:
                    self._export_log(epoch, last_total_step + total_batch_idx, total_batch_idx, curr_lr,
                                     metricsContainer.calculate_average("loss"),
                                     metricsContainer.calculate_average("acc"),
                                     metricsContainer.calculate_average("train_time"))

            if total_batch_idx % self.train_config.display_interval == 0:
                self._export_log(epoch, last_total_step + total_batch_idx, total_batch_idx, self.lr_scheduler.lr,
                                 metricsContainer.calculate_average("loss"),
                                 metricsContainer.calculate_average("acc"),
                                 metricsContainer.calculate_average("train_time"))
        return total_batch_idx + last_total_step

    def eval(self, epoch, type_list, args_list):
        """
        :param data: decide which dataset is used to eval
        :param epoch: int
        :param type_list: list[str]
        :param args_list: list[dict]
        :return:
        """
        self.model.eval()
        for eval_type, eval_args in zip(type_list, args_list):
            if eval_type == "question_answer":
                val_result = question_answer(self.model, self.val_loader, "val", **eval_args)
                test_result = question_answer(self.model, self.test_loader, "test", **eval_args)
                result = {
                    "val_accuracy": val_result,
                    "test_accuracy": test_result
                }
                self._print_metrics(epoch, result)
            else:
                raise NotImplementedError

    def train(self):
        total_step = 0
        for epoch in range(1, self.train_config["max_epoch"]):
            saved_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))
            total_step = self._train_one_epoch(epoch, total_step)
            self._save_model(saved_path)
            print("\nStart To Validate & Test")
            sys.stdout.flush()
            self.eval(epoch, self.test_config["type_list"], self.test_config["args_list"])
            print('=' * 60)
        print('-' * 120)
        print('Done.')

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.config,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        print('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        print('load model from {}, num_updates {}.'.format(path, self.num_updates))
