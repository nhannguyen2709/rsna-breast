import torch
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


class Predictor:
    def __init__(self, cls_names=("breast"), device="cuda", fp16=False):
        exp = get_exp("YOLOX/yolox_rsna.py")
        model = exp.get_model()
        if device == "cuda":
            model.cuda()
        if fp16:
            model = model.half()
        model.eval()
        ckpt_file = "YOLOX/YOLOX_outputs/yolox_rsna/best_ckpt.pth"
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)

        self.device = device
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.cls_names = cls_names

    def __call__(self, imgs, ratios):
        imgs = imgs.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(imgs)
        outputs = postprocess(
            outputs,
            self.num_classes,
            self.confthre,
            self.nmsthre,
        )
        outputs = self.rescale(outputs, ratios)
        return outputs

    def rescale(self, outputs, ratios):
        rescaled_outputs = []
        for output, ratio in zip(outputs, ratios):
            if output is None:
                rescaled_outputs.append(())
            else:
                output = output.cpu()
                bboxes = output[:, 0:4]
                # preprocessing: resize
                bboxes /= ratio
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]
                rescaled_outputs.append((bboxes, scores, cls))
        return rescaled_outputs
