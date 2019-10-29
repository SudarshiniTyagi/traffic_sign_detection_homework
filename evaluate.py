from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

from data import initialize_data # data.py in the same folder
from resnet_model import Net
from google_net import Net as Net2
from resnet_model_2 import Net as Net3
from stn import Net as Net4

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model1', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--model2', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--model3', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--model4', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--model5', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()



state_dict_1 = torch.load(args.model1)
model_1 = Net()
model_1.load_state_dict(state_dict_1)
model_1.eval()
model_1.to("cuda:0")

state_dict_2 = torch.load(args.model2)
model_2 = Net2()
model_2.load_state_dict(state_dict_2)
model_2.eval()
model_2.to("cuda:0")

state_dict_3 = torch.load(args.model3)
model_3 = Net3()
model_3.load_state_dict(state_dict_3)
model_3.eval()
model_3.to("cuda:0")

state_dict_4 = torch.load(args.model4)
model_4 = Net4()
model_4.load_state_dict(state_dict_4)
model_4.eval()
model_4.to("cuda:0")

state_dict_5 = torch.load(args.model5)
model_5 = Net3()
model_5.load_state_dict(state_dict_5)
model_5.eval()
model_5.to("cuda:0")




from data import data_transforms, data_transforms2

test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")
for f in tqdm(os.listdir(test_dir)):
    if 'ppm' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data2 = data_transforms2(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        data = Variable(data, volatile=True).to("cuda:0")
        data2 = data2.view(1, data2.size(0), data2.size(1), data2.size(2))
        data2 = Variable(data2, volatile=True).to("cuda:0")
        
        output_1 = model_1(data)
        output_2 = model_2(data)
        output_3 = model_3(data)
        output_4 = model_4(data2)
        output_5 = model_5(data)
        output = (output_1+output_2+output_3+output_4+output_5)/5
        pred = output.data.max(1, keepdim=True)[1]

        file_id = f[0:5]
        output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2018/')
        


