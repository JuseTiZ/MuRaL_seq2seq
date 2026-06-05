import numpy as np


class Observer:
    def update(self, **kwargs):
        raise NotImplementedError


class LossMinor(Observer):
    """Tracks running loss. Train script reads current_loss for progress lines."""

    def __init__(self, out_after_n_batch=500):
        self.loss = 0.0
        self.sample_number = 0
        self.train_time = 0.0
        self.loss_tasks = None
        self.counter = 0
        self.out_after_n_batch = out_after_n_batch
        self._loss_window = []
        self._window_size = 20

    @property
    def current_loss(self):
        if not self._loss_window:
            return 0.0
        return sum(self._loss_window) / len(self._loss_window)

    def reset(self):
        self.loss = 0.0
        self.sample_number = 0
        self.train_time = 0.0
        self.loss_tasks = None
        self.counter = 0

    def out_minor(self, mode):
        mean_loss = self.loss / max(self.sample_number, 1)
        minor_dict = {
            f'{mode}_mean_loss': mean_loss,
            f'{mode}_mean_time_each_batch': self.train_time / max(self.sample_number, 1),
            f'{mode}_sample_number': self.sample_number,
            f'{mode}_total_loss': self.loss,
        }
        if self.loss_tasks is not None:
            for i, task_loss in enumerate(self.loss_tasks):
                minor_dict[f'{mode}_mean_loss_task_{i}'] = task_loss / max(self.sample_number, 1)
        self.reset()
        return minor_dict

    def update(self, **kwargs):
        loss = kwargs.get('loss')
        loss_tasks = kwargs.get('loss_tasks')
        sample_number = kwargs.get('sample_number')
        epoch_finish = kwargs.get('epoch_finish')
        train_time = kwargs.get('train_time')
        mode = kwargs.get('mode')

        if loss is not None:
            self.loss += loss
            self.sample_number += sample_number

            self._loss_window.append(loss)
            if len(self._loss_window) > self._window_size:
                self._loss_window.pop(0)

            self.counter += 1

        if loss_tasks is not None:
            if self.loss_tasks is None:
                self.loss_tasks = np.zeros_like(loss_tasks)
            self.loss_tasks += loss_tasks

        if train_time is not None:
            self.train_time += train_time

        if epoch_finish is not None:
            return self.out_minor(mode)


class GradMinor(Observer):
    """Prints gradient norms periodically."""

    def __init__(self, out_after_n_batch=500):
        self.counter = 0
        self.out_after_n_batch = out_after_n_batch

    def update(self, **kwargs):
        model = kwargs.get('model')
        if model is not None:
            self.counter += 1
            if self.counter >= self.out_after_n_batch:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                print(f"[train] gradient norm: {total_norm:.4f}")
                self.counter = 0
