import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2
import time
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('/scratch/jmeng18/V2X-SIM/sweeps/CAM_id_0_1/scene_1_000006.jpg')
resize_img = cv2.resize(raw_img, (256,704))
mask = np.zeros(img.shape[:2],dtype='unit8')
center = (img.shape[1]//2, img.shape[0]//2)
radius = 100
cv2.circle(mask, center, radius, 255, -1)
masked_img = cv2.bitwise_and(resize_img, resize_img, mask=mask)
#mask_img = 
start_time = time.time()
depth = model.infer_image(masked_img) # HxW raw depth map in numpy
end_time = time.time()
print('The inference time is',end_time-start_time)
plt.figure()
plt.imshow(resize_img)
plt.axis('off')
plt.show()

#img_rgb = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
