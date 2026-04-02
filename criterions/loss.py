import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS
from ipdb import set_trace


@CRITERIONS.register('UNIFORM')
@CRITERIONS.register('NONUNIFORM')
class OadLoss(nn.Module):

    def __init__(self, cfg, reduction='mean'):
        super(OadLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['data']['num_classes']
        self.loss = self.end_loss
        self.cfg = cfg
        self.loss_mode = cfg['model']['loss']

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']

        if 'NONUNIFORM' in self.loss_mode:
            logits = logits[:, -1, :].contiguous()
            target = target[:, -1, :].contiguous()
        elif 'UNIFORM' in self.loss_mode:
            logits = logits.contiguous()
            target = target.contiguous()
        else:
            raise NotImplementedError(f"Invalid loss : {self.loss_mode}!")

        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss.
        logits: (B, K) target: (B, K)
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=-1)  # B
        # w = torch.linspace(0, 1, output.shape[1]).unsqueeze(0).to(output.device)
        # w = w**3
        # w = torch.exp(w)
        # w = w.repeat(output.shape[0], 1)
        # output *= w
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, target):
        return self.loss(out_dict, target)


@CRITERIONS.register('KLD')
class KL_loss(nn.Module):
    def __init__(self, cfg):
        super(KL_loss, self).__init__()

    def forward(self, x_mu, log_var_2):
        return 0.5 * torch.sum((x_mu) ** 2 + torch.exp(log_var_2) - log_var_2 - 1, dim=-1)


@CRITERIONS.register('RECON')
class MSEloss(nn.Module):
    def __init__(self, cfg):
        super(MSEloss, self).__init__()

    def forward(self, x, target):
        assert x.shape == target.shape, 'Inconsitent dimension!'
        return F.mse_loss(x, target, reduction='mean')


@CRITERIONS.register('TemporalInfoNCE')
class OadInfonceLoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(OadInfonceLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['data']['num_classes']
        self.loss = self.end_loss

    def end_loss(self, l_pos, l_neg):
        """
        :param l_pos: [16, 8]    [bs, tws]
        :param l_neg: [16, 65536]
        :param labels: [16, 65536+8]
        :return:
        """
        s_pos = torch.exp(l_pos)
        s_neg = torch.exp(l_neg)

        sum_s_pos = torch.sum(s_pos, dim=-1)
        sum_s_neg = torch.sum(s_neg, dim=-1)

        infonce_loss = - torch.log(sum_s_pos / (sum_s_pos + sum_s_neg))

        if self.reduction == 'mean':
            loss = torch.mean(infonce_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(infonce_loss)

        return loss

    def forward(self, out_dict, target):
        return self.loss(out_dict, target)


@CRITERIONS.register('TRNMce')
class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)
            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)


@CRITERIONS.register('MCE')
class MultipCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output


@CRITERIONS.register('OADTRMce')
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        losses = [
            'labels_encoder',
            'labels_decoder',
        ]
        self.num_classes = cfg['data']['num_classes']
        self.classification_x_loss_coef = cfg['oadtr']['classification_x_loss_coef']
        self.classification_h_loss_coef = cfg['oadtr']['classification_h_loss_coef']
        self.similar_loss_coef = cfg['oadtr']['similar_loss_coef']
        self.weight_dict = {
            'labels_encoder': self.classification_h_loss_coef,
            'labels_decoder': cfg['oadtr']['classification_pred_loss_coef'],
            'labels_x0': self.classification_x_loss_coef,
            'labels_xt': self.classification_x_loss_coef,
            'distance': self.similar_loss_coef,
        }
        self.losses = losses
        self.ignore_index = -1
        self.margin = cfg['oadtr']['margin']
        self.size_average = True
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def loss_labels(self, input, targets, name):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        #
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(outputs, targets, ignore_index=21)
        target = targets.float()
        # logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * self.logsoftmax(input[:, notice_index]), 1)
            if output.sum() == 0:  # 全为 ignore 类
                loss_ce = torch.tensor(0.).to(input.device).type_as(target)
            else:
                loss_ce = torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * self.logsoftmax(input), 1)
            if self.size_average:
                loss_ce = torch.mean(output)
            else:
                loss_ce = torch.sum(output)
        if torch.isnan(loss_ce).sum() > 0:
            set_trace()
        losses = {name: loss_ce}

        return losses

    def loss_labels_decoder(self, input, targets, name):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        #
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(outputs, targets, ignore_index=21)
        target = targets.float()
        # logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        ignore_index = -1  # -1 改为21 更好一点
        if ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * self.logsoftmax(input[:, notice_index]), 1)
            if output.sum() == 0:  # 全为 ignore 类
                loss_ce = torch.tensor(0.).to(input.device).type_as(target)
            else:
                loss_ce = torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * self.logsoftmax(input), 1)
            if self.size_average:
                loss_ce = torch.mean(output)
            else:
                loss_ce = torch.sum(output)
        if torch.isnan(loss_ce).sum() > 0:
            set_trace()
        losses = {name: loss_ce}

        return losses

    def contrastive_loss(self, output, label, name):
        """
        Contrastive loss function.
        Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        output1, output2 = output
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1. - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        if torch.isnan(loss_contrastive).sum() > 0:
            set_trace()
        losses = {name: loss_contrastive.double()}
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels_encoder': self.loss_labels,
            'labels_decoder': self.loss_labels_decoder,
            'labels_x0': self.loss_labels,
            'labels_xt': self.loss_labels,
            'distance': self.contrastive_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, name=loss)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs[loss], targets[loss]))

        return losses


@CRITERIONS.register('ANTICIPATION')
class OadAntLoss(nn.Module):

    def __init__(self, cfg, reduction='sum'):
        super(OadAntLoss, self).__init__()
        self.reduction = reduction
        self.loss = self.anticipation_loss
        self.num_classes = cfg['data']['num_classes']

    def anticipation_loss(self, out_dict, target, ant_target):
        anticipation_logits = out_dict['anticipation_logits']
        pred_anticipation_logits = anticipation_logits[:, -1, :, :].contiguous().view(-1, self.num_classes)
        anticipation_logit_targets = ant_target.view(-1, self.num_classes)
        ant_loss = self.mlce_loss(pred_anticipation_logits, anticipation_logit_targets)
        return ant_loss

    def ce_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:, -1, :].contiguous()
        target = target[:, -1, :].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss.
        logits: (B, K) target: (B, K)
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1)  # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)

        return loss

    def forward(self, out_dict, target, ant_target):
        return self.loss(out_dict, target, ant_target)


@CRITERIONS.register('TemporalContrastive')
class TemporalContrastiveLoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(TemporalContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.temperature = cfg.get('contrastive', {}).get('temperature', 0.07)
        self.negative_samples = cfg.get('contrastive', {}).get('negative_samples', 16)
        self.window_size = cfg['data']['window_size']

    def forward(self, current_features, past_features, current_labels, past_labels):
        """
        对比学习损失：当前帧与过去帧的对比
        :param current_features: [B, T, D] 当前帧特征
        :param past_features: [B, T, D] 过去帧特征
        :param current_labels: [B, T, num_classes] 当前帧标签
        :param past_labels: [B, T, num_classes] 过去帧标签
        """
        B, T, D = current_features.shape

        # 计算当前帧和过去帧的相似度
        current_norm = F.normalize(current_features, dim=-1)  # [B, T, D]
        past_norm = F.normalize(past_features, dim=-1)  # [B, T, D]

        # 计算相似度矩阵 [B, T, T]
        similarity = torch.bmm(current_norm, past_norm.transpose(1, 2)) / self.temperature

        # 创建标签掩码：相同动作类别的帧为正样本
        current_action = torch.argmax(current_labels, dim=-1)  # [B, T]
        past_action = torch.argmax(past_labels, dim=-1)  # [B, T]

        # 创建正样本掩码 [B, T, T]
        positive_mask = (current_action.unsqueeze(-1) == past_action.unsqueeze(1)) & \
                        (current_action.unsqueeze(-1) != 0)  # 排除背景类

        # 创建负样本掩码
        negative_mask = ~positive_mask

        # 计算对比损失
        exp_sim = torch.exp(similarity)

        # 正样本损失
        positive_sim = similarity * positive_mask.float()
        positive_loss = -torch.log(exp_sim * positive_mask.float() + 1e-8)
        positive_loss = positive_loss * positive_mask.float()

        # 负样本损失
        negative_sim = similarity * negative_mask.float()
        negative_loss = torch.log(exp_sim * negative_mask.float() + 1e-8)
        negative_loss = negative_loss * negative_mask.float()

        # 计算总损失
        total_loss = positive_loss.sum() + negative_loss.sum()

        if self.reduction == 'mean':
            total_loss = total_loss / (positive_mask.sum() + negative_mask.sum() + 1e-8)

        return total_loss


@CRITERIONS.register('FutureGuidedContrastive')
class FutureGuidedContrastiveLoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(FutureGuidedContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.temperature = cfg.get('contrastive', {}).get('temperature', 0.07)
        self.future_weight = cfg.get('contrastive', {}).get('future_weight', 0.5)

    def forward(self, current_features, past_features, future_features,
                current_labels, past_labels, future_labels):
        """
        未来帧引导的对比学习损失
        :param current_features: [B, T, D] 当前帧特征
        :param past_features: [B, T, D] 过去帧特征
        :param future_features: [B, T, D] 未来帧特征
        :param current_labels: [B, T, num_classes] 当前帧标签
        :param past_labels: [B, T, num_classes] 过去帧标签
        :param future_labels: [B, T, num_classes] 未来帧标签
        """
        B, T, D = current_features.shape

        # 标准化特征
        current_norm = F.normalize(current_features, dim=-1)
        past_norm = F.normalize(past_features, dim=-1)
        future_norm = F.normalize(future_features, dim=-1)

        # 当前帧与过去帧的对比
        current_past_sim = torch.bmm(current_norm, past_norm.transpose(1, 2)) / self.temperature

        # 当前帧与未来帧的对比
        current_future_sim = torch.bmm(current_norm, future_norm.transpose(1, 2)) / self.temperature

        # 创建标签掩码
        current_action = torch.argmax(current_labels, dim=-1)
        past_action = torch.argmax(past_labels, dim=-1)
        future_action = torch.argmax(future_labels, dim=-1)

        # 正样本掩码
        past_positive = (current_action.unsqueeze(-1) == past_action.unsqueeze(1)) & \
                        (current_action.unsqueeze(-1) != 0)
        future_positive = (current_action.unsqueeze(-1) == future_action.unsqueeze(1)) & \
                          (current_action.unsqueeze(-1) != 0)

        # 负样本掩码
        past_negative = ~past_positive
        future_negative = ~future_positive

        # 计算损失
        def compute_contrastive_loss(similarity, positive_mask, negative_mask):
            exp_sim = torch.exp(similarity)
            positive_loss = -torch.log(exp_sim * positive_mask.float() + 1e-8) * positive_mask.float()
            negative_loss = torch.log(exp_sim * negative_mask.float() + 1e-8) * negative_mask.float()
            return positive_loss.sum() + negative_loss.sum()

        # 过去帧对比损失
        past_loss = compute_contrastive_loss(current_past_sim, past_positive, past_negative)

        # 未来帧对比损失
        future_loss = compute_contrastive_loss(current_future_sim, future_positive, future_negative)

        # 总损失
        total_loss = past_loss + self.future_weight * future_loss

        if self.reduction == 'mean':
            total_mask_sum = (past_positive.sum() + past_negative.sum() +
                              future_positive.sum() + future_negative.sum() + 1e-8)
            total_loss = total_loss / total_mask_sum

        return total_loss


@CRITERIONS.register('AnticipationLoss')
class AnticipationLoss(nn.Module):
    """
    未来动作预测损失函数
    借鉴MROADA的设计，对预测的未来K步动作计算损失

    支持两种模式：
    1. last_frame_only=True: 只计算最后一帧的未来预测损失（在线场景，推荐）
    2. last_frame_only=False: 计算所有帧的未来预测损失（离线场景）
    """

    def __init__(self, cfg, reduction='mean'):
        super(AnticipationLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['data']['num_classes']
        self.anticipation_length = cfg.get('anticipation', {}).get('length', 5)
        self.decay_factor = cfg.get('anticipation', {}).get('decay_factor', 0.9)
        # 借鉴MROADA：在线场景只计算最后一帧
        self.last_frame_only = cfg.get('anticipation', {}).get('last_frame_only', True)

    def forward(self, anticipation_logits, labels):
        """
        计算未来预测损失
        :param anticipation_logits: [B, S, K, num_classes] 未来K步的预测logits
        :param labels: [B, S, num_classes] 当前帧的标签（需要完整序列）
        :return: 未来预测损失
        """
        B, S_logits, K, C = anticipation_logits.shape
        B_labels, S_labels, C_labels = labels.shape

        # 保存原始标签序列，用于构造未来标签
        original_labels = labels

        if self.last_frame_only:
            # 修正的策略：只计算有真实未来标签的帧
            # 排除最后 K 帧（它们没有足够的未来标签）
            # 这样可以避免 padding 导致的损失为0
            valid_length = max(1, S_logits - K)  # 至少保留1帧
            anticipation_logits = anticipation_logits[:, -valid_length:, :, :]
            # 注意：仍然保持 labels 完整

        # 构造未来标签：将标签序列向左移动k步
        future_labels = []
        for k in range(1, K + 1):
            if k < S_labels:
                # 有足够的未来标签
                shifted = torch.cat([
                    original_labels[:, k:, :],  # 未来k步的标签
                    torch.zeros(B, k, C, dtype=labels.dtype, device=labels.device)  # 末尾padding
                ], dim=1)
            else:
                # 预测步数超过序列长度，全部padding
                shifted = torch.zeros(B, S_labels, C, dtype=labels.dtype, device=labels.device)

            if self.last_frame_only:
                # 只取有效范围的未来标签（排除会是padding的部分）
                valid_length = max(1, S_labels - K)
                shifted = shifted[:, -valid_length:, :]  # [B, valid_length, C]

            future_labels.append(shifted)

        future_labels = torch.stack(future_labels, dim=2)  # [B, S, K, C]

        # 计算每个未来步的损失
        total_loss = 0
        for k in range(K):
            k_logits = anticipation_logits[:, :, k, :].contiguous()  # [B, S, C]
            k_labels = future_labels[:, :, k, :].contiguous()  # [B, S, C]

            # 多标签交叉熵损失（借鉴MROADA的mlce_loss实现）
            k_loss = self.mlce_loss(k_logits.view(-1, C), k_labels.view(-1, C))

            # 添加衰减因子：越远的未来权重越小
            decay = self.decay_factor ** k
            total_loss += decay * k_loss

        # 平均损失
        total_loss = total_loss / K

        return total_loss

    def mlce_loss(self, logits, target):
        """
        多标签交叉熵损失（与MROADA保持一致）
        :param logits: [N, C] 预测logits
        :param target: [N, C] 目标标签
        """
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=-1)  # [N]

        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        else:
            loss = output

        return loss
