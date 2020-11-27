import json
import trt_pose.coco
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)

import trt_pose.models
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

import torch
WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

import torch2trt
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

import cv2
import torchvision.transforms as transforms
import PIL.Image
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

from jetcam.csi_camera import CSICamera
camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)
camera.running = True

from jetcam.utils import bgr8_to_jpeg
import ipywidgets
from IPython.display import display
image_w = ipywidgets.Image(format='jpeg')

def execute(change):
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
execute({'new': camera.value})
camera.observe(execute, names='value')

from jetbot import Robot
robot = Robot()

import time
def stop(change):
    robot.stop()
    
def step_forward(change):
    robot.forward(0.4)
    time.sleep(0.5)
    robot.stop()

def step_backward(change):
    robot.backward(0.4)
    time.sleep(0.5)
    robot.stop()

def step_left(change):
    robot.left(0.3)
    time.sleep(0.5)
    robot.stop()

def step_right(change):
    robot.right(0.3)
    time.sleep(0.5)
    robot.stop()
    
import ipywidgets.widgets as widgets
# create buttons
button_layout = widgets.Layout(width='100px', height='80px', align_self='center')
stop_button = widgets.Button(description='stop', button_style='danger', layout=button_layout)
forward_button = widgets.Button(description='forward', layout=button_layout)
backward_button = widgets.Button(description='backward', layout=button_layout)
left_button = widgets.Button(description='left', layout=button_layout)
right_button = widgets.Button(description='right', layout=button_layout)

# link buttons to actions
stop_button.on_click(stop)
forward_button.on_click(step_backward)
backward_button.on_click(step_forward)
left_button.on_click(step_left)
right_button.on_click(step_right)

# display buttons
middle_box = widgets.HBox([left_button, stop_button, right_button], layout=widgets.Layout(align_self='center'))
controls_box = widgets.VBox([forward_button, middle_box, backward_button])
display(controls_box)

display(image_w)
