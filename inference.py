import net
import torch
import os
from face_alignment import align
import numpy as np


adaface_models = {
	'ir_18':"pretrained/adaface_ir18_vgg2.ckpt",
	'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
	'ir_101':"pretrained/adaface_ir101_ms1mv2.ckpt",
	'ir_101_v3':"pretrained/adaface_ir101_ms1mv3.ckpt",
	'ir_18_Fine':"pretrained/IR18Last.ckpt",
	'ir_50_Fine':"pretrained/IR50Last.ckpt",
	'ir_101_Fine':"pretrained/IR101last.ckpt",
	'ir_18_Fine_RIGHT':"pretrained/IR18Last_RIGHT.ckpt",
	'ir_50_Fine_RIGHT':"pretrained/IR50Last_RIGHT.ckpt",
	'ir_101_Fine_RIGHT':"pretrained/IR101Last_RIGHT.ckpt",
	'ir_18_Fine_BGR':"pretrained/IR18LastBGR.ckpt",
	'ir_50_Fine_BGR':"pretrained/IR50LastBGR.ckpt",
	'ir_101_Fine_BGR':"pretrained/IR101LastBGR.ckpt"}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_50')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = 'face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)
    

