# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------
import math

import torch
import torch.nn.functional as F
import torchvision.ops as ops


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def mask_iou(mask1, mask2):
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1[:, :, :] * mask2[:, :, :]).sum(-1).sum(-1)
    union = (mask1[:, :, :] + mask2[:, :, :] - mask1[:, :, :] * mask2[:, :, :]).sum(-1).sum(-1)

    return (intersection + 1e-6) / (union + 1e-6)


def mask_nms(seg_masks, scores, category_ids, nms_thr=0.5):
    n_samples = len(scores)
    if n_samples == 0:
        return []
    keep = [True for i in range(n_samples)]
    seg_masks = seg_masks.sigmoid() > 0.5

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        # label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]

            iou = mask_iou(mask_i, mask_j)[0]
            if iou > nms_thr:
                keep[j] = False
    return keep


class IDOL_Tracker_PosSim(object):

    def __init__(self,
                 nms_thr_pre=0.7,
                 nms_thr_post=0.3,
                 init_score_thr=0.2,
                 addnew_score_thr=0.5,
                 obj_score_thr=0.1,
                 match_score_thr=0.5,
                 memo_tracklet_frames=10,
                 memo_backdrop_frames=1,
                 memo_momentum=0.5,
                 nms_conf_thr=0.5,
                 nms_backdrop_iou_thr=0.5,
                 nms_class_iou_thr=0.7,
                 with_cats=True,
                 match_metric='bisoftmax',
                 long_match=False,
                 frame_weight=False,
                 temporal_weight=False,
                 memory_len=10,
                 pos_sim=False):
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.memory_len = memory_len
        self.temporal_weight = temporal_weight
        self.long_match = long_match
        self.frame_weight = frame_weight
        self.nms_thr_pre = nms_thr_pre
        self.nms_thr_post = nms_thr_post
        self.init_score_thr = init_score_thr
        self.addnew_score_thr = addnew_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine']
        self.match_metric = match_metric

        self.pos_sim = pos_sim
        self.center_dist_thr = 0

        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []

        self.image_sizes = None
        self.ori_size = None

    @property
    def empty(self):
        return False if self.tracklets else True

    def cal_mask_center(self, mask):
        output_h, output_w = mask.shape[-2:]
        pred_mask_i = F.interpolate(mask[None, :, :], size=(output_h * 4, output_w * 4),
                                    mode="bilinear", align_corners=False).sigmoid()
        pred_mask_i = pred_mask_i[:, :self.image_sizes[0], :self.image_sizes[1]]  # crop the padding area
        pred_mask_i = (F.interpolate(pred_mask_i, size=(self.ori_size[0], self.ori_size[1]), mode='nearest') > 0.5)[
            0, 0].cpu()  # resize to ori video size
        rows, cols = torch.where(pred_mask_i)
        if rows.nelement() == 0 or cols.nelement() == 0:
            return None, None
        x_min, y_min = torch.min(cols), torch.min(rows)  # x, y 축 각각의 최소 좌표 계산
        x_max, y_max = torch.max(cols), torch.max(rows)  # x, y 축 각각의 최대 좌표 계산
        x_center = (x_min + x_max) / 2  # x 좌표 중심점 계산
        y_center = (y_min + y_max) / 2  # y 좌표 중심점 계산
        return x_center, y_center

    def update_memo(self, ids, bboxes, embeds, labels, frame_id, masks_center):
        # masks_center : [(x_center, y_center), (x_center, y_center), ...]

        tracklet_inds = ids > -1

        masks_center = torch.tensor(masks_center).to(bboxes)

        # update memo
        for id, bbox, embed, label, mask_center in zip(ids[tracklet_inds],
                                                       bboxes[tracklet_inds],
                                                       embeds[tracklet_inds],
                                                       labels[tracklet_inds],
                                                       masks_center[tracklet_inds]):
            id = int(id)

            # if mask_center[0] == -1:
            #     print()

            if id in self.tracklets.keys():
                velocity = (bbox - self.tracklets[id]['bbox']) / (
                        frame_id - self.tracklets[id]['last_frame'])
                self.tracklets[id]['bbox'] = bbox

                self.tracklets[id]['long_score'].append(bbox[-1])
                self.tracklets[id]['embed'] = (
                                                      1 - self.memo_momentum
                                              ) * self.tracklets[id]['embed'] + self.memo_momentum * embed
                self.tracklets[id]['long_embed'].append(embed)
                # self.tracklets[id]['long_embed'].append(self.tracklets[id]['embed'])
                self.tracklets[id]['last_frame'] = frame_id
                self.tracklets[id]['label'] = label
                self.tracklets[id]['velocity'] = (
                                                         self.tracklets[id]['velocity'] *
                                                         self.tracklets[id]['acc_frame'] + velocity) / (
                                                         self.tracklets[id]['acc_frame'] + 1)
                self.tracklets[id]['acc_frame'] += 1
                self.tracklets[id]['exist_frame'] += 1

                if self.pos_sim:
                    if mask_center[0] != -1:
                        self.tracklets[id]['mask_center'] = mask_center

            else:
                if self.pos_sim:
                    self.tracklets[id] = dict(
                        bbox=bbox,
                        mask_center=mask_center,
                        embed=embed,
                        long_embed=[embed],
                        long_score=[bbox[-1]],
                        label=label,
                        last_frame=frame_id,
                        velocity=torch.zeros_like(bbox),
                        acc_frame=0,
                        exist_frame=1)
                else:
                    self.tracklets[id] = dict(
                        bbox=bbox,
                        embed=embed,
                        long_embed=[embed],
                        long_score=[bbox[-1]],
                        label=label,
                        last_frame=frame_id,
                        velocity=torch.zeros_like(bbox),
                        acc_frame=0,
                        exist_frame=1)

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
            if len(v['long_embed']) > self.memory_len:
                v['long_embed'].pop(0)
            if len(v['long_score']) > self.memory_len:
                v['long_score'].pop(0)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_vs = []
        memo_long_embeds = []
        memo_long_score = []
        memo_exist_frame = []
        memo_masks_center = []

        for k, v in self.tracklets.items():
            memo_bboxes.append(v['bbox'][None, :])
            # memo_embeds.append(v['embed'][None, :])
            memo_masks_center.append(v['mask_center'][None, :])
            if self.long_match:
                weights = torch.stack(v['long_score'])
                if self.temporal_weight:
                    length = len(weights)
                    temporal_weight = torch.range(0.0, 1, 1 / length)[1:].to(weights)
                    weights = weights + temporal_weight
                sum_embed = (torch.stack(v['long_embed']) * weights.unsqueeze(1)).sum(0) / weights.sum()
                memo_embeds.append(sum_embed[None, :])
            else:
                memo_embeds.append(v['embed'][None, :])

            memo_long_embeds.append(torch.stack(v['long_embed']))
            memo_long_score.append(torch.stack(v['long_score']))
            memo_exist_frame.append(v['exist_frame'])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_exist_frame = torch.tensor(memo_exist_frame, dtype=torch.long)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_masks_center = torch.cat(memo_masks_center, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)

        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_masks_center, memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs, memo_long_embeds, memo_long_score, memo_exist_frame

    def match(self, bboxes, labels, masks, track_feats, frame_id, indices, image_sizes, ori_size):
        self.image_sizes = image_sizes
        self.ori_size = ori_size
        # self.center_dist_thr = math.sqrt(ori_size[0] ** 2 + ori_size[1] ** 2) / 2  # diagnoal/2
        self.center_dist_thr = ori_size[1] / 2  # W/2

        embeds = track_feats

        #   mask nms
        valids = mask_nms(masks, bboxes[:, -1], None, self.nms_thr_pre)
        mask_new_indices = torch.tensor(indices)[valids].tolist()
        indices = mask_new_indices
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        masks = masks[valids]
        embeds = embeds[valids, :]

        ids = torch.full((bboxes.size(0),), -2, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_masks_center, memo_labels, memo_embeds, memo_ids,
             memo_vs, memo_long_embeds, memo_long_score, memo_exist_frame) = self.memo
            # print(memo_embeds.shape)

            mask_center_list = []

            memo_exist_frame = memo_exist_frame.to(memo_embeds)
            memo_ids = memo_ids.to(memo_embeds)
            if self.match_metric == 'longrang':
                feats = torch.mm(embeds, memo_embeds.t())
            elif self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError
            # for i in range(bboxes.size(0)):
            for i, bbox in enumerate(bboxes):
                if self.frame_weight:
                    non_backs = (memo_ids > -1) & (scores[i, :] > 0.5)
                    if (scores[i, non_backs] > 0.5).sum() > 1:
                        wighted_scores = scores.clone()
                        frame_weight = memo_exist_frame[scores[i, :][memo_ids > -1] > 0.5]
                        wighted_scores[i, non_backs] = wighted_scores[i, non_backs] * frame_weight
                        wighted_scores[i, ~non_backs] = wighted_scores[i, ~non_backs] * frame_weight.mean()
                        conf, memo_ind = torch.max(wighted_scores[i, :], dim=0)
                    else:
                        conf, memo_ind = torch.max(scores[i, :], dim=0)
                else:
                    conf, memo_ind = torch.max(scores[i, :], dim=0)

                id = memo_ids[memo_ind]
                x_center, y_center = self.cal_mask_center(masks[i])  # 주어진 마스크의 중심점 계산
                if conf > self.match_score_thr:  # f(i,j)>0.5인 경우, 메모리 뱅크 속 마스크 중심점과 거리 계산하여 위치적인 유사도 값 제공
                    if x_center is None:
                        if id > -1:
                            ids[i] = id
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                            if self.pos_sim:  # mask 예측이 빈 경우
                                mask_center_list.append((-1, -1))
                            continue
                    else:
                        memo_x_center, memo_y_center = memo_masks_center[int(memo_ind)]
                        distance = math.sqrt((x_center - memo_x_center) ** 2 + (y_center - memo_y_center) ** 2)

                        if distance < self.center_dist_thr:
                            print('\t임베딩 O / 거리 유사도 O')
                            mask_center_list.append((x_center, y_center))
                        else:
                            # print('임베딩 O / 거리 X ')
                            if (ids[i] == -2) & (bbox[4] > self.addnew_score_thr).cpu():
                                mask_center_list.append((x_center, y_center))
                            else:
                                mask_center_list.append((-1, -1))
                            continue
                    if id > -1:  # 메모리뱅크에서 추적된 개체인 경우,, 해당 객체의 tracking id 부여
                        ids[i] = id
                        scores[:i, memo_ind] = 0
                        scores[i + 1:, memo_ind] = 0
                else:  # f(i,j)<0.5이지만, 예측 점수가 0.2 이상인 경우 새로운 개체로 추가
                    # print('임베딩 X / 거리 X ')
                    if (ids[i] == -2) & (bbox[4] > self.addnew_score_thr).cpu() and x_center is not None:
                        mask_center_list.append((x_center, y_center))
                    else:
                        if x_center is None:
                            mask_center_list.append((image_sizes[1] / 2, image_sizes[0] / 2))
                        else:
                            mask_center_list.append((-1, -1))
            new_inds = (ids == -2) & (bboxes[:, 4] > self.addnew_score_thr).cpu()
            num_news = new_inds.sum()
            ids[new_inds] = torch.arange(
                self.num_tracklets,
                self.num_tracklets + num_news,
                dtype=torch.long)
            self.num_tracklets += num_news

            unselected_inds = torch.nonzero(ids == -2, as_tuple=False).squeeze(1)
            mask_ious = mask_iou(masks[unselected_inds].sigmoid() > 0.5, masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            self.update_memo(ids, bboxes, embeds, labels, frame_id, mask_center_list)


        elif self.empty:
            init_inds = (ids == -2) & (bboxes[:, 4] > self.init_score_thr).cpu()
            num_news = init_inds.sum()
            ids[init_inds] = torch.arange(
                self.num_tracklets,
                self.num_tracklets + num_news,
                dtype=torch.long)
            self.num_tracklets += num_news
            unselected_inds = torch.nonzero(ids == -2, as_tuple=False).squeeze(1)
            mask_ious = mask_iou(masks[unselected_inds].sigmoid() > 0.5, masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            mask_center_list = []
            for i, mask in enumerate(masks):
                x_center, y_center = self.cal_mask_center(mask)
                if x_center is not None:
                    mask_center_list.append((x_center, y_center))
                else:
                    mask_center_list.append((image_sizes[1] / 2, image_sizes[0] / 2))
            self.update_memo(ids, bboxes, embeds, labels, frame_id, mask_center_list)

        return bboxes, labels, ids, indices
