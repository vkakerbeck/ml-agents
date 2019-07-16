import torch
from torchvision import transforms
from mlagentsdev.trainers import depth_networks
import os
import PIL.Image as pil
#import matplotlib.pyplot as plt
import numpy as np

class Depth_Extractor(object):
    def __init__(self, depth_model_path):
        self.depth_model_path = depth_model_path
        self.encoder, self.depth_decoder = self.init_model('otc_mixed')

    def init_model(self,model):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model_path = os.path.join(self.depth_model_path, model)#XX make general
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        encoder = depth_networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        depth_decoder = depth_networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_decoder.to(device)
        depth_decoder.eval()

        return encoder, depth_decoder

    def compute_depths(self,img, manipulation):
        img = pil.fromarray(img.astype('uint8'), 'RGB')
        original_size = img.size

        if manipulation == 'resize':
            img = img.resize((160, 160))
        else:
            if original_size == (168, 168):
                img = img.crop((4, 4, 164, 164))
            else:
                raise Exception("Invalid image size {}, expecting (160, 160) or (168, 168).".format(original_size))

        if torch.cuda.is_available():
            device = torch.device("cuda")

        with torch.no_grad():
            input_image = transforms.ToTensor()(img).unsqueeze(0).to(device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, original_size, mode="bilinear", align_corners=False)
            depths = disp_resized.squeeze().cpu().numpy()

        return np.expand_dims(depths, axis=2)

#E = Depth_Extractor('C:/Users/vivia/Desktop/ml-agents-dev/depth_models/')
#d = E.compute_depths('image.jpg','crop')
#d = compute_depths('image.jpg','crop')
#plt.imshow(d)
#plt.show()
