# Derived from the supplied FIPT experiments and AdaBrain-Bench/LaBraM/CBraMod.
# See THIRD_PARTY_NOTICES.md for upstream attribution and license information.
import json
import torch

def get_fipt_lr_scale(name, backbone_lr_scale=1.0, head_lr_scale=1.0):
    """Set rates for existing backbone, new FIPT, and classification-head parameters."""
    parts = name.split('.')
    if parts[0] == 'module':
        parts = parts[1:]
    if parts[0] == 'task_head':
        return head_lr_scale
    backbone_roots = {'t_a_patch_block', 't_b_patch_block', 't_c_patch_block', 't_a_front_model', 't_b_front_model', 't_c_front_model', 't_a_mid_model', 't_b_mid_model', 't_c_mid_model', 't_a_rear_model', 't_b_rear_model', 't_c_rear_model'}
    new_parameters = {'psd_proj', 'ctx', 'pos_prompt_embed', 'new_pos_embed', 'head'}
    if parts[0] in backbone_roots and (not set(parts[1:]) & new_parameters):
        return backbone_lr_scale
    return 1.0

def get_parameter_groups(model, weight_decay=1e-05, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    backbone_lr_scale = kwargs.get('backbone_lr_scale', 1.0)
    head_lr_scale = kwargs.get('head_lr_scale', 1.0)
    if not 0 < backbone_lr_scale <= 1:
        raise ValueError('backbone_lr_scale must be in (0, 1].')
    if not 0 < head_lr_scale <= 1:
        raise ValueError('head_lr_scale must be in (0, 1].')
    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f'filter {name} because of the pattern {filter_n}')
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith('.bias') or name in skip_list:
            group_name = 'no_decay'
            this_weight_decay = 0.0
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = 'layer_%d_%s' % (layer_id, group_name)
        else:
            layer_id = None
        fipt_scale = get_fipt_lr_scale(name, backbone_lr_scale, head_lr_scale)
        group_name = ('backbone_' if fipt_scale != 1.0 else 'new_') + group_name
        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0
            scale *= fipt_scale
            parameter_group_names[group_name] = {'weight_decay': this_weight_decay, 'params': [], 'lr_scale': scale}
            parameter_group_vars[group_name] = {'weight_decay': this_weight_decay, 'params': [], 'lr_scale': scale}
        parameter_group_vars[group_name]['params'].append(param)
        parameter_group_names[group_name]['params'].append(name)
    print('Param groups = %s' % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def create_optimizer(args, model):
    if args.opt != 'adamw':
        raise ValueError('The supplied recipes use AdamW; other optimizers are not implemented.')
    groups = get_parameter_groups(model, args.weight_decay,
                                  backbone_lr_scale=args.backbone_lr_scale,
                                  head_lr_scale=args.head_lr_scale)
    for group in groups:
        group['lr'] = args.lr * group['lr_scale']
    kwargs = dict(lr=args.lr, weight_decay=0., eps=args.opt_eps)
    if args.opt_betas is not None:
        kwargs['betas'] = tuple(args.opt_betas)
    return torch.optim.AdamW(groups, **kwargs)
