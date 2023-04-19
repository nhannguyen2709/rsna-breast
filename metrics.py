def pfbeta_metric(labels, preds, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0.0


def pfbeta_loss(labels, preds, beta=1):
    return 1 - pfbeta_metric(labels, preds, beta)
