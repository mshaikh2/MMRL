import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from catr.models import caption
from catr.datasets import coco, utils
from catr.configuration import Config

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
args = parser.parse_args()
image_path = args.path
version = args.v

config = Config()
device = torch.device(config.device)

# # set torch hub path for model cache
# torch.hub.set_dir('catr/.cache/hub')
# if version == 'v1':
#     model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
# elif version == 'v2':
#     model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
# elif version == 'v3':
#     model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
# else:
#     raise NotImplementedError('Version not implemented')
# model.to(device)

model, _ = caption.build_model(config)
model.to(device)

print("Loading Checkpoint %s..." % version)
checkpoint = torch.load('catr/checkpoint_%s.pth' % version, map_location='cpu')
model.load_state_dict(checkpoint['model'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

image = Image.open(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0)
# print(image.shape)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

image = image.to(device)
caption = caption.to(device)
cap_mask = cap_mask.to(device)

@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
#         print(predictions.shape)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
#         print(predicted_id)
        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


output = evaluate()
outlist = output.cpu().tolist()
result = tokenizer.batch_decode(outlist, skip_special_tokens=True)
for rcap in result: 
    print(rcap.capitalize(), sep='\n')