import torch
import torch.nn.functional as F
import option

args = option.parse_args()
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve

torch.autograd.set_detect_anomaly(True)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        self_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        pos_mask = mask - self_mask
        neg_mask = 1 - mask

        exp_logits = torch.exp(similarity_matrix) * (1 - self_mask)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        loss = - (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        return loss.mean()
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        d = torch.cdist(x, y, p=2)
        return d

    def forward(self, feats, margin=100.0):
        bs = len(feats)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]
        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)
        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)
        a_d_min = margin - a_d_min
        a_d_min = torch.max(torch.zeros(bs // 2).cuda(), a_d_min)
        return torch.mean(n_d_max) + torch.mean(a_d_min)


# 原版
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()
        self.contrust = ContrastiveLoss()

    def forward(self, scores, feats, targets, alpha=0.01,beta=0.4):
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats)
        loss_construct = self.contrust(feats,targets)
        return loss_ce,  alpha* loss_triplet


def train(loader, model, optimizer, scheduler, device, epoch):
    with torch.set_grad_enabled(True):
        model.train()
        pred = []
        label = []
        for step, (ninput, nlabel, ainput, alabel) in tqdm(enumerate(loader)):
            input = torch.cat((ninput, ainput), 0).to(device)

            scores, feats, = model(input)
            pred = scores.cpu().detach().tolist() + pred
            labels = torch.cat((nlabel, alabel), 0).to(device)
            label = labels.cpu().detach().tolist() + label

            loss_criterion = Loss()
            loss_ce, loss_con1, loss2 = loss_criterion(scores.squeeze(), feats, labels)
            loss = loss_ce  + loss_con1 +loss2

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step_update(epoch * len(loader) + step)
        fpr, tpr, _ = roc_curve(label, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(label, pred)
        pr_auc = auc(recall, precision)
        print('train_pr_auc : ' + str(pr_auc))
        print('train_roc_auc : ' + str(roc_auc))
        return loss.item()
