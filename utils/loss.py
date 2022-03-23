
def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))

def loss_computation(logits_list, labels, losses, edges=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        # Whether to use edges as labels According to loss type.
        if loss_i.__class__.__name__ in ('BCELoss', ) and loss_i.edge_label:
            loss_list.append(losses['coef'][i] * loss_i(logits, edges))
        else:
            loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list