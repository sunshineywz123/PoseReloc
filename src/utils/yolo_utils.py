import cv2
import torch
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False,
              scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constrains
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def process_img(img0, img_size, stride, cvt_color=True):
    # Preprocess image
    assert img0 is not None, "Image not found!"

    # img = letterbox(img0, img_size, stride=stride)[0]
    img = letterbox(img0, img_size, stride=stride, scaleFill=False, auto=False)[0]

    if cvt_color:
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB
    else:
        img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float().cuda()
    img /= 255.
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img

    
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                        agnostic=False, multi_label=False, labels=()):
    """ Runs Non-Maximum Suppression (NMS) on inference results
    
    Returns:
        list of detections, on (n , 6) tensor per image [xyxy, conf, cls] 
    """
    import torchvision

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    # Settings 
    min_wh, max_wh = 2, 4096 # (pixel) minimum add maximum box width and height
    max_det = 300 # maximum number of detections per image
    max_nms = 30000 # maximum number of boxes into torchvision.ops.nms()
    redundant = True # require redundant detections
    multi_label = multi_label + (nc > 1)
    merge = False

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0 # width - height
        x = x[xc[xi]]

        # Cat apriori labesl if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc+5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0 # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else: # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh) # classes
        boxes, scores = x[:, :4] + c, x[:, 4] # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres) # NMS
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3): # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i, 4) = weights(i, n) * boxes(n, 4)
            iou = box_iou(boxes[i], boxes) > iou_thres # iou matrix
            weights = iou * scores[None] # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True) # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1] # require redundancy

        output[xi] = x[i]

    return output     


def box_iou(box1, box2):
    """
    Return intersection-over-union of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the N*M matrix containing the pairwise
                            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N, M) = (rb(N, M, 2) - lt(N, M, 2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return iter / (area1[:, None] + area2 - inter) # iou = iter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y11, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2 # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2 # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2 # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2 # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy boudning boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1]) # x1
    boxes[:, 1].clamp_(0, img_shape[0]) # y1
    boxes[:, 2].clamp_(0, img_shape[1]) # x2
    boxes[:, 3].clamp_(0, img_shape[0]) # y2


class YOLOWarper:
    def __init__(self, cfg):
        yolo_model_path = cfg.model.detection_model_path
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=yolo_model_path)
        self.stride = int(self.yolo_model.stride.max())
        self.yolo_det_size = 640
        self.process_img = process_img
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords

    def prepare_data(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = img.astype(np.float32)
        img_size = img.shape[:2]
        img = img[None]
        img /= 255.

        inp = torch.Tensor(img)[None].cuda()
        return inp, np.array(img_size)[None]

    def __call__(self, img_path, K):
        from src.utils import data_utils

        img = cv2.imread(img_path)
        inp_yolo = self.process_img(img, self.yolo_det_size, self.stride)
        pred_yolo = self.yolo_model(inp_yolo)[0]

        pred_yolo = self.non_max_suppression(pred_yolo)

        if pred_yolo[0].shape[0] == 0:  # No obj is detected
            return None

        for i, det in enumerate(pred_yolo):
            det[:, :4] = self.scale_coords(inp_yolo.shape[2:], det[:, :4], img.shape).round()
            box = det[0, :4].cpu().numpy().astype(np.int)

        x0, y0, x1, y1 = box
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
        image_crop = data_utils.get_image_crop_resize(img, box, resize_shape)

        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([512, 512])
        K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)

        inp_spp, img_size = self.prepare_data(image_crop)
        return inp_spp, image_crop, img_size, K_crop
