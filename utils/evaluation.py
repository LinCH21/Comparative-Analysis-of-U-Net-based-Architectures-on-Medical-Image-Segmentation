import torch
import pandas as pd
import os

def compute_hd95(pred, target):
    # Get the coordinates of the boundary points
    pred_points = torch.nonzero(pred, as_tuple=False)
    target_points = torch.nonzero(target, as_tuple=False)

    # Compute pairwise distances between all boundary points
    dists = torch.cdist(pred_points.float(), target_points.float(), p=2)

    # For each point in A, find the minimum distance to a point in B, and vice versa
    min_dists_A_to_B = torch.min(dists, dim=1)[0]
    min_dists_B_to_A = torch.min(dists, dim=0)[0]

    # Compute the 95th percentile of the distances
    hd95_A_to_B = torch.quantile(min_dists_A_to_B, 0.95)
    hd95_B_to_A = torch.quantile(min_dists_B_to_A, 0.95)

    # Return the maximum of the two
    return torch.max(hd95_A_to_B, hd95_B_to_A).item()

def compute_metrics(predictions, targets):
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate TP, TN, FP, FN
    TP = (predictions * targets).sum().float()
    TN = ((1 - predictions) * (1 - targets)).sum().float()
    FP = (predictions * (1 - targets)).sum().float()
    FN = ((1 - predictions) * targets).sum().float()

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    hausdorff_dist_95 = compute_hd95(predictions, targets)

    return dice.item(), accuracy.item(), precision.item(), hausdorff_dist_95

def evaluation(args, model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.model_path, args.model_name + "_checkpoint.pt"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scores = {
        "index": [],
        "dice": [],
        "accuracy": [],
        "precision": [],
        "hd95": []
    }
    with torch.no_grad():
        for batch in test_dataloader:
            test_idx = batch["index"]
            test_image = batch["data"]
            test_label = batch["label"]
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            test_pred = model(test_image)
            pred_label = torch.argmax(test_pred, 1)
            cur_dice, cur_acc, cur_pre, cur_hd = compute_metrics(pred_label, test_label)
            scores["index"].append(test_idx)
            scores["dice"].append(cur_dice)
            scores["accuracy"].append(cur_acc)
            scores["precision"].append(cur_pre)
            scores["hd95"].append(cur_hd)
    df = pd.DataFrame(scores)
    return df

if __name__ == "__main__":
    pred = torch.randn(1, 1, 224, 224).cuda()
    label = torch.randn(1, 1, 224, 224).cuda()
    print(compute_metrics(pred, label))
