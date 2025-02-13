# -*- coding: utf-8 -*-
import torch
from shapely.geometry import Polygon
import itertools


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def eight_coords_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax)
    representation for comparison to 8-coords form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax) form of boxes.
    """
    tmp = torch.cat((boxes[:, :2] - boxes[:, 2:]/2, 
                    boxes[:, :2] + boxes[:, 2:]/2), 1)
    xmin = tmp[:, 0]
    ymin = tmp[:, 1]
    xmax = tmp[:, 2]
    ymax = tmp[:, 3]

    return torch.cat((xmin.unsqueeze(1), ymin.unsqueeze(1), xmax.unsqueeze(1), ymin.unsqueeze(1),
                      xmax.unsqueeze(1), ymax.unsqueeze(1), xmin.unsqueeze(1), ymax.unsqueeze(1)),
                     dim=1)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard_quadrilateral(box_a, box_b):
    """Compute the jaccard overlap of two sets of quadrilaterals. The
    jaccard overlap is simply the intersection over union of two
    boxes. Here we operate on ground truth boxes and default boxes.
    
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects, 8]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 8]
    Return:
        IoU(jaccard) overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]

    """
    # print("a shape: {}".format(box_a.shape))
    # print("b shape: {}".format(box_b.shape))
    quadri_a = [Polygon(ql.reshape((-1,2))) for ql in box_a]
    quadri_b = [Polygon(ql.reshape((-1,2))) for ql in box_b]

    box_product = itertools.product(quadri_a, quadri_b)
    intersections_list = [a.intersection(b).area for (a, b) in box_product]
    intersections_tensor = torch.tensor(intersections_list, dtype=torch.float).reshape(box_a.size(0), box_b.size(0))
    area_a = torch.tensor([qa.area for qa in quadri_a], dtype=torch.float).unsqueeze(1).expand_as(intersections_tensor)
    area_b = torch.tensor([qb.area for qb in quadri_b], dtype=torch.float).unsqueeze(0).expand_as(intersections_tensor)

    union = area_a + area_b - intersections_tensor
    return intersections_tensor / union

    
def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match_quadrilaterals(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors, 8].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard_quadrilateral(
        truths,
        eight_coords_form(priors)
    )
    # (Bipartite Matching)
    # downwards: gt; horizontal: priors
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    pos = torch.where(conf>0)
    loc = encode_quadrilaterals(matches, priors, pos, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior



def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

    
def define_orient(matched, horizontal_rectangle):
    '''
    确定多边形四个角的顺序，采用欧几里得距离，求最小的排列
    :param matched: [num_priors, 8], point-form
    :param horizontal_rectangle: [num_priors, 8], point-form
    :return:
    '''
    # 1234-2341-3412-4123
    distance_min_1 = torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                   torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                   torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                   torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 6:8]) ** 2, dim=1) # (num_priors)

    distance_min_2 = torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                     torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                     torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                     torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 6:8]) ** 2, dim=1)  # (num_priors)

    distance_min_3 = torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                     torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                     torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                     torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 6:8]) ** 2, dim=1)  # (num_priors)

    distance_min_4 = torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                     torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                     torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                     torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 6:8]) ** 2, dim=1)  # (num_priors)

    origin = torch.cat((distance_min_1.unsqueeze(1),
                        distance_min_2.unsqueeze(1),
                        distance_min_3.unsqueeze(1),
                        distance_min_4.unsqueeze(1)), 1) # (num_priors, 4)

    origin_indices = origin.min(dim=1).indices  # (num_priors)
    for i, box in enumerate(matched):
        if origin_indices[i] == 1:
            index = [2, 3, 4, 5, 6, 7, 0, 1]
            matched[i] = matched[i][index]
        if origin_indices[i] == 2:
            index = [4, 5, 6, 7, 0, 1, 2, 3]
            matched[i] = matched[i][index]
        if origin_indices[i] == 3:
            index = [6, 7, 0, 1, 2, 3, 4, 5]
            matched[i] = matched[i][index]

    return matched


def encode_quadrilaterals(matched, priors, pos, variances):
    """

    Args:
        matched: (tensor) Coords of ground truth for each prior in concatenated 4 * (x, y) coords
            Shape: [num_priors, 8].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 8]

    """
    # todo: 查看matched 和 priors的值是多少（绝对坐标值还是相对w、h的值）
    x0 = matched[:, 0:7:2].min(-1, keepdim=True).values
    y0 = matched[:, 1:8:2].min(-1, keepdim=True).values
    x1 = matched[:, 0:7:2].max(-1, keepdim=True).values
    y1 = matched[:, 1:8:2].max(-1, keepdim=True).values
    horizontal_rectangle = torch.cat((x0, y0, x1, y0, x1, y1, x0, y1), 1)
    matched[pos] = define_orient(matched[pos], horizontal_rectangle[pos])

    priors_8coords = eight_coords_form(priors)
    
    diff = matched - priors_8coords
    diff /= torch.cat((priors[:, 2:], priors[:, 2:], priors[:, 2:], priors[:, 2:]), dim=1)

    return diff

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    return variances[1] * (loc + eight_coords_form(priors))

    # boxes = torch.cat((
    #     priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    #     priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # boxes[:, :2] -= boxes[:, 2:] / 2
    # boxes[:, 2:] += boxes[:, :2]
    # return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,8].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
