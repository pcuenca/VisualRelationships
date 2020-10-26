import random
import numpy as np

import torch
import torchvision.transforms as transforms

from param import args
from speaker import Speaker
from data_infer import InferenceDataset, TorchDataset

# Set the seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Image Transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
    normalize
])

# Change workers
#if args.img_type == 'pixel':
#    args.workers = 1    # In Memory Loading
#elif args.img_type == 'feat':
#    args.workers = 2

# Loading Dataset
def get_tuple(ds_name, split, task='speaker', shuffle=True, drop_last=True):
    dataset = InferenceDataset(ds_name, split, args.train)
    torch_ds = TorchDataset(dataset, task, max_length=args.max_input,
        img0_transform=img_transform, img1_transform=img_transform
    )
    print("The size of data split %s is %d" % (split, len(torch_ds)))
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True,
        drop_last=drop_last)
    return dataset, torch_ds, loader

assert args.train == 'speaker_inference'

test_tuple = get_tuple(args.dataset, 'test', shuffle=False, drop_last=False)
speaker = Speaker(test_tuple[0])    # [0] is the dataset

if args.load is not None:
    print("Load speaker from %s." % args.load)
    speaker.load(args.load)

result = speaker.inference(test_tuple)
print("Inference:")
print("Got results for %d items" % len(result))
import json
json.dump(result, open("inference_result.json", 'w'))


