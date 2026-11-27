"""Unified single-device FIPT training entry point."""
import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from engine_for_finetuning import train_one_epoch, evaluate, configure_binary_class_balance
from util.eegdatasets import load_splits
from util.optim_factory import create_optimizer
from util.model_storage import pretrained_cache, share_frozen_backbone, parameter_counts
from util.utils import NativeScalerWithGradNormCount, cosine_scheduler

ROOT = Path(__file__).resolve().parent

def load_recipe(name):
    path = ROOT / 'configs' / (name + '.json')
    if not path.is_file() or path.parent != ROOT / 'configs':
        raise ValueError(f'Unknown recipe: {name}')
    return json.loads(path.read_text(encoding='utf-8'))

def model_args(recipe, **overrides):
    values = recipe['defaults'].copy()
    values.update(nb_classes=1, backbone_lr_scale=values.get('backbone_lr_scale', 1.0),
                  head_lr_scale=values.get('head_lr_scale', 1.0),
                  class_balance=values.get('class_balance', False),
                  binary_smoothing=values.get('binary_smoothing', 0.0),
                  fusion2_gate_init=values.get('fusion2_gate_init', 0.01),
                  pretrained='', from_scratch=False, share_frozen=True, defer_sharing=False)
    values.update(overrides)
    return SimpleNamespace(**values)

def build_model(recipe, args, ch_names):
    module = importlib.import_module('models.' + recipe['recipe'])
    with pretrained_cache(args):
        model = module.build_model(args, ch_names, recipe['num_t'])
    if args.share_frozen and not args.defer_sharing:
        share_frozen_backbone(model)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['DEAP', 'HCI', 'MER', 'REFED'], default='REFED')
    parser.add_argument('--model_name', '--backbone', choices=['LaBraM', 'CBraMod'], default='LaBraM')
    parser.add_argument('--target', choices=['arousal', 'valence'], default='valence')
    parser.add_argument('--list', action='store_true', help='List the 16 recipes and exit')
    parser.add_argument('--data-root', type=Path, help='Directory containing train.json, val.json, test.json')
    parser.add_argument('--pretrained', type=Path, help='Trusted backbone checkpoint')
    parser.add_argument('--from-scratch', action='store_true', help='Explicit diagnostic mode; no pretrained weights')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'finetuning_results')
    parser.add_argument('--seeds', type=int, nargs='+', help='Seed IDs; default: 0 1 2 3 4 for every recipe')
    parser.add_argument('--no-share-frozen', action='store_true', help='Use original duplicated backbone storage for comparison')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--update_freq', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--backbone_lr_scale', type=float)
    parser.add_argument('--selection-metric', choices=['balanced_accuracy'], default='balanced_accuracy',
                        help='Save the best checkpoint by validation balanced accuracy')
    parser.add_argument('--resume', type=Path, help='Resume checkpoint-last.pth for a single --seeds ID')
    parser.add_argument('--evaluate', type=Path, help='Evaluate a trusted full FIPT checkpoint; no training')
    parser.add_argument('--deterministic', action='store_true', help='Request deterministic PyTorch operations')
    return parser.parse_args()

def main():
    cli = parse_args()
    if cli.list:
        print((ROOT / 'configs/index.json').read_text(encoding='utf-8'))
        return
    recipe = load_recipe(f'{cli.dataset.lower()}_{cli.model_name.lower()}_{cli.target}')
    args = model_args(recipe)
    for key in ('epochs','batch_size','update_freq','lr','weight_decay','warmup_epochs','backbone_lr_scale'):
        if getattr(cli,key) is not None:
            setattr(args,key,getattr(cli,key))
    args.share_frozen = not cli.no_share_frozen
    args.defer_sharing = True
    args.device = cli.device
    args.num_workers = cli.num_workers
    args.from_scratch = cli.from_scratch or cli.evaluate is not None or cli.resume is not None
    args.pretrained = str((cli.pretrained or ROOT / 'checkpoints' /
                         ('labram-base.pth' if cli.model_name == 'LaBraM' else 'pretrained_weights.pth')).resolve())
    if not args.from_scratch and not Path(args.pretrained).is_file():
        raise FileNotFoundError(f'Backbone checkpoint missing: {args.pretrained}; set --pretrained')
    if min(args.epochs,args.batch_size,args.update_freq) < 1:
        raise ValueError('epochs, batch_size and update_freq must be positive')
    if not 0 <= args.warmup_epochs <= args.epochs:
        raise ValueError('warmup_epochs must be between 0 and epochs')
    if args.layer_decay != 1.0 or args.finetune_mod != 'full':
        raise ValueError('This release preserves the supplied FIPT tuning masks (full recipe mode)')
    device = torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = not cli.deterministic
    torch.use_deterministic_algorithms(cli.deterministic)
    config = json.loads((ROOT/'dataset_config/Classification.json').read_text())
    data_root = cli.data_root or ROOT / config[cli.dataset]['root']['cross']
    train, val, test = load_splits(data_root, recipe)
    if args.nb_classes == 1:
        configure_binary_class_balance(args, train)
    metric = cli.selection_metric or recipe['selection_metric']
    metrics = ['accuracy', 'balanced_accuracy', 'pr_auc', 'roc_auc']
    seeds = cli.seeds if cli.seeds is not None else recipe['seeds']
    if len(seeds)!=len(set(seeds)):
        raise ValueError('Seed IDs must be unique')
    if cli.evaluate:
        seeds = seeds[:1]
    if cli.resume and (cli.evaluate or len(seeds)!=1):
        raise ValueError('--resume requires exactly one --seeds ID and cannot be combined with --evaluate')
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        args.seed = seed
        run_dir = cli.output_dir / recipe['recipe'] / f'seed_{seed}'
        if not cli.evaluate and not cli.resume and (run_dir/'log.txt').exists():
            raise FileExistsError(f'{run_dir} already has results; choose another --output_dir')
        run_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(recipe, args, train.get_ch_names()).to(device)
        test_loader = DataLoader(test,batch_size=args.batch_size,sampler=SequentialSampler(test),
                                 num_workers=args.num_workers,pin_memory=device.type=='cuda')
        if cli.evaluate:
            checkpoint=torch.load(cli.evaluate,map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint.get('model',checkpoint),strict=True)
            if args.share_frozen:
                share_frozen_backbone(model)
            result=evaluate(args,test_loader,model,device,metrics=metrics)
            print(json.dumps(result,indent=2))
            (run_dir/'evaluation.json').write_text(json.dumps(result,indent=2))
            continue
        # Matches the supplied single-process DistributedSampler (fixed epoch=0).
        train_loader = DataLoader(train,batch_size=args.batch_size,
            sampler=DistributedSampler(train,num_replicas=1,rank=0,shuffle=True),
            num_workers=args.num_workers,pin_memory=device.type=='cuda',drop_last=True)
        val_loader = DataLoader(val,batch_size=args.batch_size,sampler=SequentialSampler(val),
                                num_workers=args.num_workers,pin_memory=device.type=='cuda')
        steps=len(train)//(args.batch_size*args.update_freq)
        if steps==0:
            raise ValueError('Training split smaller than one effective batch; reduce --batch_size or --update_freq')
        if args.warmup_steps>args.epochs*steps:
            raise ValueError('Warmup exceeds training length')
        if args.share_frozen and not cli.resume:
            share_frozen_backbone(model)
        optimizer=create_optimizer(args,model)
        scaler=NativeScalerWithGradNormCount(enabled=device.type=='cuda')
        lrs=cosine_scheduler(args.lr,args.min_lr,args.epochs,steps,
                             warmup_epochs=args.warmup_epochs,warmup_steps=args.warmup_steps)
        wds=cosine_scheduler(args.weight_decay,args.weight_decay if args.weight_decay_end is None else args.weight_decay_end,
                             args.epochs,steps)
        metadata=dict(recipe=recipe['recipe'],source_folder=recipe['source_folder'],seed=seed,
                      source_selection_metric=recipe['source_selection_metric'],
                      recipe_sha256=hashlib.sha256((ROOT/'configs'/(recipe['recipe']+'.json')).read_bytes()).hexdigest(),
                      model_sha256=hashlib.sha256((ROOT/'models'/(recipe['recipe']+'.py')).read_bytes()).hexdigest(),
                      selection_metric=metric,requested_seeds=seeds,data_root=str(Path(data_root).resolve()),args=vars(args),
                      torch_version=torch.__version__,split_sizes=[len(train),len(val),len(test)],
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      parameter_counts=parameter_counts(model))
        if not cli.resume:
            (run_dir/'run_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
        best=-float('inf')
        start_epoch=0
        if cli.resume:
            checkpoint=torch.load(cli.resume,map_location='cpu',weights_only=False)
            if checkpoint.get('recipe')!=recipe['recipe'] or checkpoint.get('seed')!=seed:
                raise ValueError('Resume checkpoint recipe/seed does not match the request')
            if checkpoint.get('selection_metric')!=metric:
                raise ValueError('Resume must use the original validation selection criterion')
            for key in ('epochs','batch_size','update_freq','lr','weight_decay','warmup_epochs','backbone_lr_scale'):
                if checkpoint['args'][key]!=getattr(args,key):
                    raise ValueError(f'Resume configuration changed: {key}')
            records=[json.loads(line) for line in (run_dir/'log.txt').read_text().splitlines() if line.strip()]
            if not records or records[-1]['epoch']!=checkpoint['epoch']:
                raise ValueError('Resume checkpoint and output log must end at the same epoch')
            model.load_state_dict(checkpoint['model'],strict=True)
            if args.share_frozen:
                share_frozen_backbone(model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            best=checkpoint['best_score'];start_epoch=checkpoint['epoch']+1
            torch.set_rng_state(checkpoint['torch_rng'])
            np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
            if device.type=='cuda' and checkpoint['cuda_rng'] is not None:
                torch.cuda.set_rng_state(checkpoint['cuda_rng'],device)
        start=time.time()
        for epoch in range(start_epoch,args.epochs):
            tr=train_one_epoch(args,model,train_loader,optimizer,device,epoch,scaler,
                start_steps=epoch*steps,lr_schedule_values=lrs,wd_schedule_values=wds,
                num_training_steps_per_epoch=steps,ch_names=train.get_ch_names())
            va=evaluate(args,val_loader,model,device,header='Val:',metrics=metrics)
            te=evaluate(args,test_loader,model,device,header='Test:',metrics=metrics)
            record={'epoch':epoch,**{'train_'+k:v for k,v in tr.items()},
                    **{'val_'+k:v for k,v in va.items()},**{'test_'+k:v for k,v in te.items()}}
            if not np.isfinite(va[metric]):
                raise FloatingPointError(f'Validation {metric} is non-finite')
            if va[metric]>best:
                best=va[metric]
                torch.save(dict(model=model.state_dict(),epoch=epoch,args=vars(args),recipe=recipe['recipe'],
                                selection_metric=metric,val_metrics=va,test_metrics=te),run_dir/'checkpoint-best.pth')
                (run_dir/'best.json').write_text(json.dumps(record,indent=2),encoding='utf8')
            with (run_dir/'log.txt').open('a',encoding='utf8') as f:
                f.write(json.dumps(record)+'\n')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                epoch=epoch,best_score=best,recipe=recipe['recipe'],seed=seed,args=vars(args),selection_metric=metric,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None),run_dir/'checkpoint-last.pth')
        (run_dir/'completed.json').write_text(json.dumps(dict(epochs=args.epochs,elapsed_seconds=time.time()-start)))
        del model,optimizer,scaler
        if device.type=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':
    main()
