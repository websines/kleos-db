def acc(pred, gold):
    return round(sum(p == g for p, g in zip(pred, gold)) / len(gold), 3)

def binary_report(p, y, thr=0.5):
    n = len(y)
    a = sum((pi >= thr) == yi for pi, yi in zip(p, y)) / n
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    # ECE, 10 equal-width bins on confidence of the predicted class
    bins = [[] for _ in range(10)]
    for pi, yi in zip(p, y):
        conf = max(pi, 1 - pi); corr = (pi >= thr) == yi
        bins[min(int(conf * 10), 9)].append((conf, corr))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(k for _, k in b) / len(b)) for b in bins if b)
    # best achievable accuracy over thresholds (how separable the scores are)
    best = max(sum((pi >= t) == yi for pi, yi in zip(p, y)) / n for t in sorted(set(p)) + [1.01])
    # AUROC
    pos = [pi for pi, yi in zip(p, y) if yi]; neg = [pi for pi, yi in zip(p, y) if not yi]
    auc = sum((a_ > b_) + 0.5 * (a_ == b_) for a_ in pos for b_ in neg) / (len(pos) * len(neg))
    return {"acc": round(a, 3), "brier": round(brier, 3), "ece": round(ece, 3), "auroc": round(auc, 3), "oracle_thr_acc": round(best, 3)}
