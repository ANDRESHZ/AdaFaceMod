import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    print(image_path)
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        width, height = img.size
        #41 WI 53.3 HE
        left = width*(0.4-0.205)
        top = height*(0.4-0.283)
        right = width*(0.6+0.205)
        bottom = height*(0.6+0.25)
        img2=img.crop((left, top, right, bottom))
        face=img2.resize((112,112))
        #print('Face detection Failed due to error.')
        #print(e)
        #face = None

    return face


