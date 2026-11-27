"""Checkpoint I/O and storage sharing; model computations remain task-local."""
from contextlib import contextmanager
from collections.abc import Mapping
import copy
import re
import torch


def _copy_containers(value):
    # Loading code may pop checkpoint keys. Share immutable source tensors, not dictionaries.
    if isinstance(value, Mapping):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = _copy_containers(item)
        return result
    return value


@contextmanager
def pretrained_cache(args):
    previous = getattr(args, '_fipt_pretrained_cache', None)
    args._fipt_pretrained_cache = {}
    try:
        yield
    finally:
        if previous is None:
            delattr(args, '_fipt_pretrained_cache')
        else:
            args._fipt_pretrained_cache = previous


def load_pretrained(args):
    cache = getattr(args, '_fipt_pretrained_cache', None)
    if cache is None:
        return torch.load(args.pretrained, map_location='cpu', weights_only=False)
    if args.pretrained not in cache:
        cache[args.pretrained] = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    return _copy_containers(cache[args.pretrained])


def _role(name):
    labram = re.match(r't_[abc]_(patch_block|front_model|mid_model|rear_model)\.(.+)', name)
    if labram:
        return 'labram', labram[1], labram[2]
    cbramod = re.match(r'model_(patch|patch_fft|trans)_t_[abc]\.(.+)', name)
    if cbramod:
        # The temporal patch weights duplicated inside transformer wrappers have identical keys.
        return 'cbramod', cbramod[2]
    return None


def parameter_counts(model):
    return {
        'logical_parameters': sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False)),
        'unique_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def share_frozen_backbone(model):
    """Alias only identical, frozen, corresponding backbone parameters.

    Call AFTER device transfer and loading weights. Independent LayerNorm/prompt/fusion/
    head parameters remain independent. No averaging, approximation or tensor conversion.
    State-dictionary keys remain unchanged. Forward operations and dropout calls are retained.
    """
    before = parameter_counts(model)
    candidates = {}
    aliases = []
    for name, parameter in list(model.named_parameters(remove_duplicate=False)):
        role = _role(name)
        if role is None or parameter.requires_grad:
            continue
        owner_name, attr = name.rsplit('.', 1)
        owner = model.get_submodule(owner_name)
        if getattr(owner, 'doWeightNorm', False) or getattr(owner, 'max_norm', None) is not None:
            continue
        key = (role, tuple(parameter.shape), parameter.dtype, parameter.device)
        matches = candidates.setdefault(key, [])
        canonical = next(((n, p) for n, p in matches if torch.equal(parameter, p)), None)
        if canonical is None:
            matches.append((name, parameter))
        else:
            owner._parameters[attr] = canonical[1]
            aliases.append({'alias': name, 'canonical': canonical[0], 'elements': parameter.numel()})
    after = parameter_counts(model)
    assert before['trainable_parameters'] == after['trainable_parameters']
    report = dict(before=before, after=after, shared_frozen_elements=before['unique_parameters']-after['unique_parameters'],
                  aliases=aliases)
    model.storage_report = report
    if not getattr(model, '_fipt_storage_hooks', False):
        # PyTorch releases before 1.13 expose only the private pre-hook
        # registration method.  Register with ``with_module=True`` so both
        # APIs call the same hook signature used below.
        pre_hook = getattr(model, 'register_load_state_dict_pre_hook', None)
        if pre_hook is None:
            model._register_load_state_dict_pre_hook(_unshare_before_load, with_module=True)
        else:
            pre_hook(_unshare_before_load)
        model.register_load_state_dict_post_hook(_reshare_after_load)
        model._fipt_storage_hooks = True
    return report


def _unshare_before_load(model, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # An external checkpoint can contain different frozen weights for different modalities.
    # Restore independent destinations before copying so no alias can overwrite another.
    for item in model.storage_report['aliases']:
        name=item['alias']
        parent, attr=name.rsplit('.',1)
        owner=model.get_submodule(parent)
        value=owner._parameters[attr]
        owner._parameters[attr]=torch.nn.Parameter(value.detach().clone(),requires_grad=False)


def _reshare_after_load(model, incompatible_keys):
    share_frozen_backbone(model)
