"""Teacher-forced joint token/dependency/action/target/value objectives."""
import torch
from torch.nn import functional as F


def joint_loss(outputs, tokens, graph, actions, targets, values, weights=None):
    weights = dict(token=1., graph=1., action=1., target=1., value=1.) | (weights or {})
    if not graph['action_mask'].gather(1,actions[:,None]).all():
        raise ValueError('teacher action is invalid for current graph')
    target_mask=graph['target_action_mask'][torch.arange(actions.shape[0],device=actions.device),actions]
    if not target_mask.gather(1,targets[:,None]).all():
        raise ValueError('teacher target is inactive')
    active = graph['active'][:,:,None] & graph['active'][:,None,:]
    losses = dict(token=F.cross_entropy(outputs['logits'][:,:-1].reshape(-1,outputs['logits'].shape[-1]), tokens[:,1:].reshape(-1)),
                  graph=F.cross_entropy(outputs['dependency_logits'][active], graph['dependency_labels'][active]),
                  action=F.cross_entropy(outputs['action_logits'],actions),
                  target=F.cross_entropy(outputs['target_logits'][torch.arange(actions.shape[0],device=actions.device),actions],targets),
                  value=F.mse_loss(outputs['value'], values.float()))
    return sum(weights[key]*value for key,value in losses.items()), losses
