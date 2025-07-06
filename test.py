from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)
image = cv2.imread("data/dataset/golfballs/many/0ef52bd4947d50a9eeb7740b7bc1c7a.jpg")  # 需转换为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)
input_point = np.array([[250, 250]])  # 图像上的一个点坐标
input_label = np.array([0])           # 1表示前景点
masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
masks = masks.astype(np.uint8) * 255  # 将掩码转换为0-255范围的uint8格式
cv2.imshow("Mask", masks[0])
cv2.waitKey(0)
cv2.destroyAllWindows()