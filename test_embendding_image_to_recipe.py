import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from trijoint import im2recipe
import json
import pickle
from PIL import Image
from args import get_parser

parser = get_parser()
opts = parser.parse_args()

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def test_oneimage(image_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    im = Image.open(image_path).convert('RGB')
    transform_test_image = transforms.Compose([
                         transforms.Resize(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(256),  # we get only the center of that rescaled
                         transforms.ToTensor(),
                         normalize,
                     ])
    im = transform_test_image(im)
    with torch.no_grad():
        #im = torch.autograd.Variable(im.cuda())
        im = torch.autograd.Variable(im)

    im = torch.unsqueeze(im,0)


    with open('./data/index_from_id.json', 'r') as f:
        index_from_id = json.load(f)

    with open('./data/classes1M.pkl', 'rb') as f:
        class_dict = pickle.load(f)
        classindex = pickle.load(f)
    #checkpoint = torch.load(opts.model_path,encoding='latin1')
    checkpoint = torch.load(opts.model_path,encoding='latin1',map_location='cpu')


    model = im2recipe()
    #device = torch.device('cpu')
    #model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    #model.cuda()
    #print (checkpoint['state_dict'])
    '''
    check_point
    optimizer
    state_dict
    valtrack
    best_val
    epoch
    freeVision
    curr_val
    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        #print (k)
        name = k.replace(".module","") # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(checkpoint['state_dict'])
    #model.cpu()
    model.eval()
    output = model.visionMLP(im)
    output = output.view(output.size(0), -1)
    #print (output.shape,type(output))
    output = model.visual_embedding(output)
    output = norm(output)
    output = model.semantic_branch(output)

    #print ("outputsize",output.size())
    #im = torch.Tensor(1, 3, 224, 224)
    #im=resnet(im)
    #im = torch.tensor(1, 3, 224, 224)
    #print(resnet(im))
    fuck = nn.functional.softmax(output, 1)
    #print (fuck)
    maxk = max((1, ))
    _, pred = fuck.topk(maxk, 1, True, True)

    pred = pred.data.cpu().numpy()

    batch_pre = []
    for ii in range(len(pred)):
        now_pre = []
        for j in pred[ii]:
            now_pre.append(classindex[j])

        batch_pre.append(now_pre)

    for ii in range(len(pred)):
        for j in range(len(batch_pre[ii])):
            pred_name = batch_pre[ii][j]
            bianhao = pred[ii][j]

            '''
            print (pred_name_top1,target_recipe_top1)
            print(index_from_id[target_recipe_top1]["title"])
            print(index_from_id[target_recipe_top1]["ingredients"])
            print(index_from_id[target_recipe_top1]["instructions"])
            print("......................................")
            '''
            for kk, key in enumerate(index_from_id):
                if class_dict[key] == bianhao:
                    print(index_from_id[key]["title"])
                    print (index_from_id[key]["ingredients"])
                    print (index_from_id[key]["instructions"])
                    print("************************************")
            print("......................................")

if __name__ == '__main__':
    test_oneimage("./data/test.jpg")

