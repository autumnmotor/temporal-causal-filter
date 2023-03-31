import os
import cv2
import numpy as np
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def has_mps() -> bool:
    if not getattr(torch, 'has_mps', False):
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("mps") if has_mps() else device


torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Temporal causal filter for Sequencial Images')

# filepath
parser.add_argument('--inputdir', default="/Volumes/SSD/macstudio/batch_i2i_output")
parser.add_argument('--outputdir',default="/Volumes/SSD/macstudio/batch_i2i_output_fx")

# parameter for rife
parser.add_argument('--reverb_depth',type=int,default=1)
parser.add_argument('--scale',type=float,default=1.0)

# use AI function
parser.add_argument('--rife',type=bool,default=True) # for lpf and reverb
parser.add_argument('--realesrgan',type=bool,default=True) # for deblur

args = parser.parse_args()

rife_model_path="train_log/"
realesrgan_model_path="weight/realesr-animevideov3.pth"

if not os.path.exists(realesrgan_model_path):
    print("download realesrgan_model")
    os.system(f"wget 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth' -O {realesrgan_model_path}")
else:
    print("cache found(realesrgan_model)")

if not os.path.exists(rife_model_path):
    print("download rife_model")
    os.system("wget 'https://drive.google.com/uc?export=download&id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_' -O ./tmp.zip")
    os.system("unzip ./tmp.zip -d ./")
else:
    print("cache found(rife_model)")


# restorer
upsampler = RealESRGANer(
    scale=4,
    model_path=realesrgan_model_path,
    dni_weight=0.0,
    model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device="cuda" if torch.cuda.is_available() else "mps" if has_mps() else "cpu"
    )


def load_rife_model():
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model

                model = Model()
                model.load_model(rife_model_path, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model

                model = Model()
                model.load_model(rife_model_path, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model

            model = Model()
            model.load_model(rife_model_path, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(rife_model_path, -1)
        print("Loaded ArXiv-RIFE model")
    return model

model=load_rife_model()
model.eval()
model.device()

import glob

filenames = [os.path.join(args.inputdir, x) for x in sorted(os.listdir(args.inputdir)) if not x.startswith(".")]
imgfilelist=[file for file in filenames if os.path.isfile(file)]

imgfilelist.sort()


prvs=None

reverb_img=None

def read_img(imgfilepath):
    global w,h
    img = cv2.imread(imgfilepath, cv2.IMREAD_COLOR)
    img = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    n, c, h, w = img.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img = F.pad(img, padding)

    return img


for i in range(len(imgfilelist)):

    imgfile=imgfilelist[i]
    print(imgfile)

    img = read_img(imgfile)

    next_img=img
    prvs_img=img

    try:
        next_img = read_img(imgfilelist[i+1])
    except:
        None

    try:
        prvs_img = read_img(imgfilelist[i-1 if i-1>=0 else 0])
    except:
        None

    def intp(img1,img2,to_img1_strength=0):
        if id(img1)==id(img2):
            return img1
        if img1 is None:
            return img2
        if img2 is None:
            return img1

        if args.rife:
            mid_img=model.inference(img1,img2,args.scale)

            for i in range(to_img1_strength):
                mid_img=model.inference(img1,mid_img,args.scale)

            return mid_img
        else:
            return img1

    # shallow low pass filter
#    v_img=intp(intp(img,prvs_img),intp(img,next_img))
#    mid_img=intp(img,v_img)

    v_img=intp(prvs_img,next_img)
    mid_img=intp(img,v_img)


    # shallow reverb
    if args.reverb_depth!=0:
        reverb_img=intp(mid_img,reverb_img,args.reverb_depth)
        out_img=intp(mid_img,reverb_img)
    else:
        out_img=mid_img

    write_img=(out_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    if args.realesrgan:
        write_img, _ = upsampler.enhance(write_img, outscale=1)

    cv2.imwrite(os.path.join(args.outputdir,os.path.basename(imgfile)), write_img)

print("Done")