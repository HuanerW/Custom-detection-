import os
import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchvision.io import read_image
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import os
from os import listdir
from torchvision.io import read_image
#from engine import train_one_epoch, evaluate
import utils
import os
import os
from os import listdir
import numpy as np
from skimage import io
from lxml import etree
import torch
from torch.utils.data import Dataset
#from scripts.utils import *

np.random.seed(37)
torch.manual_seed(37)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class TrainDataset(Dataset):
    def __init__(self, root_dir,transform=None, detection_transform=None):
        """
        Args:
            xml_file  (string): Path to the xml file with annotations.
            root_dir  (string): Directory with all the images.
            train     (bool): Option to output train or test data.
            data_type (string): Output data type of __getitem__(). Options: 'binary_class'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.detection_transform = detection_transform
        self.images_list=[]
        self.name_list=[]
        for images in os.listdir(root_dir):
            if (images.endswith(".png")):
                  print(images)
                  self.name_list.append(images[0:6])
                  #image2=io.imread(f"/Users/jules/downloads/Tweezers/{images}")
                  #image2=torchvision.transforms.ToTensor()(image2)
                  
                  image2=cv2.imread(f"/Users/jules/downloads/Tweezers/{images}",cv2.IMREAD_COLOR)
                  image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB).astype(np.float32)
                  image2/=255.0
          #        img = Image.open(f"/Users/jules/downloads/Tweezers/{images}").convert("RGB")
                  image2=torchvision.transforms.ToTensor()(image2)
                  
                  self.images_list.append(image2)
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self,idx):
        print("idx is")
        print(idx)
      #  if torch.is_tensor(idx):
       #     idx = idx.tolist()
    # for idx in range(len(self.images_list)):
        image = self.images_list[idx]
        name=self.name_list[idx]
        print(name)
        labels = []
       # boxes = []
        tree = ET.parse(f"/Users/jules/downloads/Tweezers/{name}.xml")
        print(tree)
        root = tree.getroot()
        target2={}
        target2['object']=[]
        for obj in root.findall('./object'):
            labels.append(1)
            o={}
            b = obj.find('bndbox')
            o['xmin'] = int(b.find('xmin').text)
            o['ymin'] = int(b.find('ymin').text)
            o['xmax'] = int(b.find('xmax').text)
            o['ymax'] = int(b.find('ymax').text)
          #  xmin=int(root.find("./object/bndbox/xmin").text)
          #  ymin=int(root.find("./object/bndbox/ymin").text)
          #  xmax=int(root.find("./object/bndbox/xmax").text)
          #  ymax=int(root.find("./object/bndbox/ymax").text)
            target2['object'].append(o)
        boxes = [[obj[f] for f in ['xmin', 'ymin', 'xmax', 'ymax']] for obj in target2['object']]
        areas= [(obj[2]-obj[0])*(obj[3]-obj[1]) for obj in boxes]
        areas=torch.as_tensor(areas, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["area"]=areas
        target["image_id"] = torch.as_tensor([idx], dtype=torch.int64)
       # target["keypoints"] = keypoints
        target["labels"] = labels
        target["iscrowd"] = iscrowd
       # target["filename"] = name
        
        if self.detection_transform:
            image, target = self.detection_transform(image, target)
        elif self.transform:
            image = self.transform(image)
        
        return image, target


train_dataset = TrainDataset('/Users/jules/downloads/Tweezers/')
#print(train_dataset)
#indices = torch.randperm(len(train_dataset)).tolist()

#dataset_train = Subset(train_dataset, indices[0:800])

def collate_fn(batch):
    return tuple(zip(*batch))
train_data_loader =DataLoader(
      train_dataset,
      batch_size=2,
      shuffle=True,
      num_workers=0,
      collate_fn=collate_fn
)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = get_device()


model = fasterrcnn_resnet50_fpn(True).to(device)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#num_classes = 1
#in_features = model.roi_heads.box_predictor.cls_score.in_features

#model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#model = model.to(device)

'''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
#num_classes =2
#in_features=model.roi_heads.box_predictor.cls_score.in_features
#model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)
'''

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 5
'''
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model,train_data_loader, device=device)
'''

iter=1
for epoch in range(num_epochs):
     for image,target in train_data_loader:
        #3 print(image)
        # print(target)
         image=list(images.to(device) for images in image)
         target =[{k:v.to(device) for k, v in t.items()} for t in target]
        # print(image)
        # print(target)
         loss_dict=model(image,target)
         losses=sum(loss for loss in loss_dict.values())
         loss_value=losses.item()
         optimizer.zero_grad()
         losses.backward()
         optimizer.step()
         
         if iter % 50 ==0:
                print(loss_value)
         iter+=1
         lr_scheduler.step()
     print(f"Epoch #{epoch} loss: {loss_value}) ")
     
torch.save(model.state_dict(),'model.pth')
torch.save({
     'epoch': epoch,
     'model_state_dict': model.state_dict(),
     'optimizer_state_dict': optimizer.state_dict()},
     'ckpt.pth'
     
)


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
from PIL import Image
def get_prediction(dataset, idx, model):
    img, _ = dataset[idx]

    model.eval()
  #  with torch.no_grad():
    prediction = model([img.to(device)])
    print("inner predict")
    print(prediction)
    return img, prediction

def get_rects(boxes):
    rect = lambda x, y, w, h: patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
    boxes=boxes.detach().numpy()
    return [rect(box[0], box[1], box[2], box[3]) for box in boxes]


def show_prediction(img, fig, ax):
    pil_image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().detach().numpy())
    print("here")
    ax.imshow(pil_image)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    print(prediction[0]['boxes'])
    for rect in get_rects(prediction[0]['boxes']):
        print("here2")
        ax.add_patch(rect)

#print(train_dataset)

predictions = [get_prediction(train_dataset, i, model) for i in range(4)]

print(predictions)

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for (img, prediction), a in zip(predictions, ax):
    show_prediction(img,fig, a)
plt.tight_layout()
'''
x=cv2.imread("/Users/jules/downloads/Test1.png",cv2.IMREAD_COLOR)
x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)
x/=255.0
        
x=torchvision.transforms.ToTensor()(x)
x2=[]
x2.append(x)
#y=[]
#y.append(x)

y=cv2.imread("/Users/jules/downloads/Test1.png",cv2.IMREAD_COLOR)
#x= read_image("/Users/jules/downloads/Test1.png")

model.load_state_dict(torch.load('./model.pth'))
model.to(device)
model.eval()
predictions=model(x2)
idx=0
boxes =predictions[idx]['boxes'].data.cpu().numpy()
print(boxes)
#drawn_boxes = draw_bounding_boxes(x, boxes[1],  colors="red")
#show(drawn_boxes)
#boxes=predictions[idx]['boxes'].data.cpu().numpy()
#print(boxes)


im=cv2.rectangle(y,(int(boxes[1][0]),int(boxes[1][1])),(int(boxes[1][2]),int(boxes[1][3])),(220,0,0),4)
cv2.imshow('Curr',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
#cv2.imshow("target", im)
#cv2.waitKey(0)
  
# closing all open windows
#cv2.destroyAllWindows()
        

image=[]
# get the path/directory
folder_dir = "/Users/jules/downloads/Tweezers/"
for images in os.listdir(folder_dir):
    
    # check if the image ends with png
    if (images.endswith(".png")):
        print(images[:6])
        image2=cv2.imread(f"/Users/jules/downloads/Tweezers/{images}",cv2.IMREAD_COLOR)
        image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB).astype(np.float32)
        image2/=255.0
        
        image2=torchvision.transforms.ToTensor()(image2)
        image.append(image2)






def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]




                                             
annotation=filelist('/Users/jules/downloads/Tweezers/','xml')
boxes=[]
labels=[]
print(annotation)
targets=[]
print("here")
root=ET.parse('/Users/jules/downloads/Tweezers/Image1.xml').getroot()
print(int(root.find("./object/bndbox/xmin").text));
for anno in annotation:
        print(ET.parse(anno))
        target={}
        root = ET.parse(anno).getroot()
        print(root)
        xmin=int(root.find("./object/bndbox/xmin").text)
        ymin=int(root.find("./object/bndbox/ymin").text)
        xmax=int(root.find("./object/bndbox/xmax").text)
        ymax=int(root.find("./object/bndbox/ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
       # boxes=[int(root.find("./object/bndbox/xmin").text),int(root.find("./object/bndbox/ymin").text),int(root.find("./object/bndbox/xmax").text),int(root.find("./object/bndbox/ymax").text)]
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels.append(1)
        labels=torch.as_tensor(labels,dtype=torch.int64)
        target["boxes"]=boxes
        target["labels"]=labels
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target["iscrowd"] = iscrowd
        targets.append(target)
     #   target['xmin'] = int(root.find("./object/bndbox/xmin").text)
     #   target['ymin'] = int(root.find("./object/bndbox/ymin").text)
     #   target['xmax'] = int(root.find("./object/bndbox/xmax").text)
     #   target['ymax'] = int(root.find("./object/bndbox/ymax").text)
     #   target["labels"] =int(1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

labels=torch.ones((boxes.shape[0]))
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
tranforms = weights.transforms()
numclasses=2
#images = list(image for image in images)
print(len(image))
print(len(targets))
output = model(image, targets)

model.eval()
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]


x=cv2.imread("/Users/jules/downloads/Image3.png",cv2.IMREAD_COLOR)
x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)
x/=255.0
        
x=torchvision.transforms.ToTensor()(x)
y=[]
y.append(x)

predictions = model(y)

y=cv2.imread("/Users/jules/downloads/Image3.png",cv2.IMREAD_COLOR)
#sample=x.cpu().numpy()
print(predictions[0]["boxes"])
boxes=predictions[0]["boxes"].data.cpu().numpy()
fig, ax = plt.subplots(1,1,figsize=(12,6))
#for box in boxes:
print(predictions[0]["boxes"][0])
im=cv2.rectangle(y,(int(predictions[0]["boxes"][0][0]),int(predictions[0]["boxes"][0][1])),(int(predictions[0]["boxes"][0][2]),int(predictions[0]["boxes"][0][3])),(220,0,0),1)
cv2.imshow("target", im)
cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()
#ax.set_axis_off()
#ax

"""

for i in range(4):
     for j in range(1):
       images=image[j]
       targetC=[{k:v.to(device) for k,v in j.items()} for j in targets]
       loss_dict=model(image,targetC)
       losses=sum(loss for loss in loss_dict.values())
       loss_value=losses.item()
       optimizer.zero_grad()
       losses.backward()
       optimizer.step()
       lr_scheduler.step()
       

torch.save(model.state_dict(),'model.pth')
torch.save({'epoch': i,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),

           },'ckpt.pth')

"""
'''
'''
x=cv2.imread("/Users/jules/downloads/Tweezers/Image1.png",cv2.IMREAD_COLOR)
x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)
x/=255.0
x=torchvision.transforms.ToTensor()(x)
model.load_state_dict(torch.load('./model.pth'))
model.to(device)
x=x.to(device)
x= x.unsqueeze_(0)
#output = model(x,targets)
'''
'''
sample=x.cpu().numpy()
boxes=output["boxes"].data.cpu().numpy()
fig, ax = plot.subplots(1,1,figsize=(12,6))
for box in boxes:
     cv2.rectangle(sample,(box[0],box[1]),(box[2],box[3]),(220,0,0),1)
ax.set_axis_off()
ax.imshow(sample)
#model.eval()
#in_features=model.roi_heads.box_predictor.cls_score.in_features
#model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

'''


'''
x=cv2.imread("/Users/jules/downloads/Tweezers/Image1.png",cv2.IMREAD_COLOR)
x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)
x/=255.0
#x=torchvision.transforms.ToTensor()(x)
#x.unsqueeze_(1)
predictions = model(x)
#print(predictions[1]["boxes"].data.cpu().numpy())


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 40


#img = tranforms(img)
#detection_outputs = model(img.unsqueeze(0), [target])
'''


