# Derived from the supplied FIPT experiments and AdaBrain-Bench/LaBraM/CBraMod.
# See THIRD_PARTY_NOTICES.md for upstream attribution and license information.
import math
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
from timm.utils import ModelEma
from timm.loss import LabelSmoothingCrossEntropy
import util.utils as utils


class BinaryLabelSmoothingLoss(torch.nn.Module):
    """Training-only binary target smoothing with optional class balancing."""

    def __init__(self, smoothing=0.0, class_weights=None):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError('binary_smoothing must be in [0, 1).')
        weights = torch.tensor(class_weights if class_weights is not None else [1.0, 1.0], dtype=torch.float32)
        if weights.shape != (2,) or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('class_weights must contain two finite positive weights.')
        self.smoothing = smoothing
        self.register_buffer('class_weights', weights)

    def forward(self, logits, targets):
        soft_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        losses = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction='none')
        weights = torch.where(targets >= 0.5, self.class_weights[1], self.class_weights[0])
        return (losses * weights).mean()


def configure_binary_class_balance(args, dataset_train):
    """Estimate binary class weights from training labels only."""
    counts = [0, 0]
    for index in range(len(dataset_train)):
        target = torch.as_tensor(dataset_train[index][1])
        if target.numel() != 1 or not torch.isfinite(target).all():
            raise ValueError('Binary training labels must be finite scalars.')
        value = target.item()
        if value not in (0, 1):
            raise ValueError('Binary training labels must be 0 or 1, got %r' % value)
        counts[int(value)] += 1
    if min(counts) == 0:
        raise ValueError('Both classes must occur in training data; counts=%r' % counts)
    args.train_class_counts = counts
    args.train_label_index = getattr(dataset_train, 'label_index', None)
    args.binary_class_weights = ([sum(counts) / (2.0 * count) for count in counts]
                                 if args.class_balance else [1.0, 1.0])
    print('Training label index:', args.train_label_index,
          'class counts:', counts, 'loss weights:', args.binary_class_weights)

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return (loss, outputs)

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, model_ema: Optional[ModelEma]=None, log_writer=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None, num_training_steps_per_epoch=None, ch_names=None):
    is_binary = args.nb_classes == 1
    if args.task_mod == 'Regression':
        criterion = torch.nn.MSELoss()
    elif args.nb_classes == 1:
        if getattr(args, 'binary_smoothing', 0.0) == 0 and not getattr(args, 'class_balance', False):
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = BinaryLabelSmoothingLoss(
                getattr(args, 'binary_smoothing', 0.0),
                getattr(args, 'binary_class_weights', None),
            ).to(device)
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    model.train(True)
    if args.finetune_mod == 'linear':
        model.main_model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    update_freq = args.update_freq
    print_freq = 20
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step
        if lr_schedule_values is not None or (wd_schedule_values is not None and data_iter_step % update_freq == 0):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group['lr'] = lr_schedule_values[it] * param_group.get('lr_scale', 1.0)
                if wd_schedule_values is not None and param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = wd_schedule_values[it]
        if args.norm_method == 'mv':
            samples = samples.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            samples = samples.float().to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        else:
            targets = targets.int().long()
        loss, output = train_class_batch(model, samples, targets, criterion)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError('Non-finite training loss: %s' % loss_value)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=is_second_order, update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict().get('scale', 1.0)
        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        if args.task_mod == 'Classification':
            if is_binary:
                class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ['accuracy'], is_binary)['accuracy']
            else:
                class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group['lr'])
            max_lr = max(max_lr, group['lr'])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group['weight_decay'] > 0:
                weight_decay_value = group['weight_decay']
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head='loss')
            if args.task_mod == 'Classification':
                log_writer.update(class_acc=class_acc, head='loss')
            log_writer.update(loss_scale=loss_scale_value, head='opt')
            log_writer.update(lr=max_lr, head='opt')
            log_writer.update(min_lr=min_lr, head='opt')
            log_writer.update(weight_decay=weight_decay_value, head='opt')
            log_writer.update(grad_norm=grad_norm, head='opt')
            log_writer.set_step()
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(args, data_loader, model, device, header='Test:', ch_names=None, metrics=['acc']):
    is_binary = args.nb_classes == 1
    if args.task_mod == 'Regression':
        criterion = torch.nn.MSELoss()
    elif is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter='  ')
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]
        if args.norm_method == 'mv':
            EEG = EEG.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            EEG = EEG.float().to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        else:
            target = target.int().long()
        loss, output = train_class_batch(model, EEG, target, criterion)
        if is_binary and args.task_mod != 'Regression':
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()
        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)
        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret
