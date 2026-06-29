import numpy as np
import skimage.transform
import onnxruntime as ort


# part of this code is based on the YoloLite infer_onnx_decoded.py script

def letterbox(im, new_size=640, color=114/255):
    h, w = im.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = skimage.transform.resize(im, (nh, nw))
    top = (new_size - nh) // 2
    bottom = top + nh
    left = (new_size - nw) // 2
    right = left + nw
    im_padded = np.full((new_size, new_size),
                         color, dtype=im_resized.dtype)
    im_padded[top:bottom, left:right] = im_resized
    return im_padded, scale, (left, top)

def nms_np(boxes, scores, iou_th=0.5, max_det=300):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= max_det:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_th]
    return np.array(keep, dtype=np.int64)



class HipDetector:
    """Hip detection model using YOLOLite model.

    This class initializes the YOLOLite model and detects candidate hip ROIs
    in the image. The best results for left and right are returned.

    The ROI is a square centered on the femoral head.

    Attributes
    ----------
    sess : ONNX inference session
    img_size : int
        size of the input image in pixels
    confidence : float
        confidence threshold used in object detection
    iou : float
        intersection-over-union threshold used in object detection
    max_detection : int
        maximum number of detections to return (YOLO leftover, not relevant here)
    """

    # YoloLite defaults
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    CLASS_NAMES = ['left', 'right']

    def __init__(self, onnx_model, img_size=640, confidence=0.25, iou=0.50, max_detections=300):
        # ort session
        sess = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
        self.sess = sess
        self.img_size = 640
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections

    def process(self, image):
        """Process an image and return hip detections.

        Returns the best detections for left and right hips, as a dictionary mapping
        the side ("left" or "right") to a dictionary with center_x, center_y, diameter,
        and score.

        Parameters
        ----------
        image : numpy array
            the input image

        Returns
        -------
        a dict of { str : dict }
            a dictionary with the best detections for left and right hips
        """
        # normalize intensities to 0-1
        intensity_offset = image.min()
        intensity_slope = image.max() - intensity_offset
        image = (image - intensity_offset) / intensity_slope

        # rescale and pad to required size
        image_letterboxed, scale, (padx, pady) = letterbox(image, self.img_size)

        # convert to three channels and apply intensity scaling from the model training
        image_letterboxed = (image_letterboxed[:, :, None] - self.MEAN[None, None, :]) / self.STD[None, None, :]

        # apply model
        in_name = self.sess.get_inputs()[0].name
        out_names = [o.name for o in self.sess.get_outputs()]  # ['boxes_xyxy','obj_logits','cls_logits']
        im = np.transpose(image_letterboxed, (2,0,1))[None]  # [1,3,H,W]
        im = im.astype(np.float32)
        boxes, obj_log, cls_log = self.sess.run(out_names, {in_name: im})

        # from here, we follow the YoloLite implementation

        # process output
        obj = 1/(1+np.exp(-obj_log[...,0]))  # [1,N]
        if cls_log.shape[-1] > 1:
            cls_sig = 1/(1+np.exp(-cls_log[0]))          # [N,C]
            confs = cls_sig.max(axis=-1)                 # [N]
            cls_id = cls_sig.argmax(axis=-1).astype(np.int64)
            scores = obj[0] * confs
        else:
            cls_id = np.zeros_like(obj[0], dtype=np.int64)
            scores = obj[0]

        # filter + per-klass NMS
        m = scores > self.confidence
        boxes_p = boxes[0][m]
        scores_p = scores[m]
        cls_p = cls_id[m]

        # postprocessing to find boxes
        final_b, final_s, final_c = [], [], []
        for c in np.unique(cls_p):
            mc = (cls_p == c)
            keep = nms_np(boxes_p[mc], scores_p[mc], self.iou, self.max_detections)
            if keep.size:
                final_b.append(boxes_p[mc][keep])
                final_s.append(scores_p[mc][keep])
                final_c.append(np.full((keep.size,), int(c), dtype=np.int64))

        if final_b:
            boxes_pad = np.concatenate(final_b, 0)
            scores_pad= np.concatenate(final_s, 0)
            classes   = np.concatenate(final_c, 0)
        else:
            boxes_pad = np.zeros((0,4), np.float32)
            scores_pad= np.zeros((0,), np.float32)
            classes   = np.zeros((0,), np.int64)

        # back-map to original
        boxes_px = boxes_pad.copy()
        boxes_px[:,[0,2]] -= padx
        boxes_px[:,[1,3]] -= pady
        boxes_px /= max(scale, 1e-6)
        h0, w0 = image.shape[:2]
        boxes_px[:,[0,2]] = np.clip(boxes_px[:,[0,2]], 0, w0-1)
        boxes_px[:,[1,3]] = np.clip(boxes_px[:,[1,3]], 0, h0-1)

        # now, select at most two hips

        # return detections
        detections = {}
        for box, score, class_idx in zip(boxes_px, scores_pad, classes):
            class_name = self.CLASS_NAMES[class_idx]
            # detections should be sorted by score, so we can take
            # the first detection for left and right and ignore any others
            if class_name not in detections:
                detections[class_name] = {
                    'center_x': np.mean(box[[0, 2]]).item(),
                    'center_y': np.mean(box[[1, 3]]).item(),
                    'diameter': np.mean(np.abs(box[[2, 3]] - box[[0, 1]])).item(),
                    'score': score.item(),
                }

        return detections



if __name__ == '__main__':
    detector = HipDetector('checkpoints/yololite_model_decoded.onnx')

    import dicom_util
    import imageio
    input_image = 'images/OAI-9763898-V00-20051017.dcm'

    if input_image.lower().endswith('.dcm'):
        _, img_pixels, _ = dicom_util.load_dicom_image(input_image)
    elif input_image.lower().endswith('.jpg'):
        img_pixels = imageio.v2.imread(input_image).astype(float)
        if img_pixels.ndim == 3:
            img_pixels = np.mean(img_pixels, axis=2)

    detections = detector.process(img_pixels)
    print(detections)
