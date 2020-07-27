'''
if you use two-stage detector, such as faster rcnn,please change the codes :
1. mmdet/models/detectors/two_stage.py

    def extract_feat(self, img):
    """Directly extract features from the backbone+neck
    """
    x_backbone = self.backbone(img)
    if self.with_neck:
        x_fpn = self.neck(x_backbone)
    return x_backbone,x_fpn

and:

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x_backbone,x_fpn = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x_fpn, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x_fpn, proposal_list, img_metas, rescale=rescale),x_backbone,x_fpn

2.mmdet/apis/inference.py

    def inference_detector(model, img):
    .......
            # forward the model
        with torch.no_grad():
            result,x_backbone,x_fpn= model(return_loss=False, rescale=True, **data)
        return result,x_backbone,x_fpn

if you use other detectors, it is easy to achieve it like this

'''



from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np
import time
import torch
import os

def main():

    config = 'work_dirs/cascade_rcnn_r18_fpn_1x_coco_frozen_stage_0/cascade_rcnn_r18_fpn_1x_coco.py'
    checkpoint = 'work_dirs/cascade_rcnn_r18_fpn_1x_coco_frozen_stage_0/epoch_12.pth'
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    img = '1.jpg'
    image = cv2.imread(img)
    height, width, channels = image.shape
    result, x_backone, x_fpn = inference_detector(model, img)

    if not os.path.exists('feature_map'):
        os.makedirs('feature_map')

    feature_index = 0
    for feature in x_backone:
        feature_index += 1
        P = torch.sigmoid(feature)
        P = P.cpu().detach().numpy()
        P = np.maximum(P, 0)
        P = (P - np.min(P)) / (np.max(P) - np.min(P))
        P = P.squeeze(0)
        print(P.shape)

        P = P[10, ...]  # 挑选一个通道
        print(P.shape)

        cam = cv2.resize(P, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        heatmap_image = np.uint8(255 * heatmap)

        cv2.imwrite('feature_map/' + 'stage_' + str(feature_index) + '_heatmap.jpg', heatmap_image)
        result = cv2.addWeighted(image, 0.8, heatmap_image, 0.3, 0)
        cv2.imwrite('feature_map/' + 'stage_' + str(feature_index) + '_result.jpg', result)

    feature_index = 1
    for feature in x_fpn:
        feature_index += 1
        P = torch.sigmoid(feature)
        P = P.cpu().detach().numpy()
        P = np.maximum(P, 0)
        P = (P - np.min(P)) / (np.max(P) - np.min(P))
        P = P.squeeze(0)
        P = P[2, ...]
        print(P.shape)
        cam = cv2.resize(P, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        heatmap_image = np.uint8(255 * heatmap)

        cv2.imwrite('feature_map/' + 'P' + str(feature_index) + '_heatmap.jpg', heatmap_image)  # 生成图像
        result = cv2.addWeighted(image, 0.8, heatmap_image, 0.4, 0)
        cv2.imwrite('feature_map/' + 'P' + str(feature_index) + '_result.jpg', result)


if __name__ == '__main__':
    main()
