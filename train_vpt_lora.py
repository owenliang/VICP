import os
import torch.nn as nn
import numpy as np
import transformers

import torch
from PIL import Image
import torch.nn.functional as F
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional
import glob

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import os
import pandas as pd
from PIL import Image
import torch

def compute_rank1_rank5_map(features, labels):
	"""
	Compute Rank-1, Rank-5 accuracy, and Mean Average Precision (mAP).
	
	:param features: Tensor of shape (N, D), where N is the number of samples and D is the feature dimension.
	:param labels: Tensor of shape (N,), containing the class labels for each sample.
	:return: rank1_acc, rank5_acc, mean_AP
	"""

	# Normalize features to unit length (useful for cosine similarity)
	features = torch.nn.functional.normalize(features, dim=1)

	# Compute similarity matrix using cosine similarity
	similarity_matrix = torch.matmul(features, features.T)  # (N, N)

	# Remove self-matching by setting diagonal values to -inf
	N = labels.size(0)
	identity_mask = torch.eye(N, dtype=torch.bool, device=features.device)
	similarity_matrix[identity_mask] = -float("inf")  # Ensures self-match is ignored

	# Sort indices based on similarity scores (descending order, most similar first)
	sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
	sorted_indices = sorted_indices[:, :-1]
	# print(sorted_indices)

	# Compute Rank-1 and Rank-5 accuracy
	rank1_correct = (labels[sorted_indices[:, 0]] == labels).float().mean()  # Top-1 match
	rank5_correct = (labels[sorted_indices[:, :5]] == labels.unsqueeze(1)).any(dim=1).float().mean()  # Top-5 match

	# Compute Mean Average Precision (mAP)
	ap_list = []
	for i in range(N):
		# Identify correct matches in the sorted list
		relevant = labels[sorted_indices[i]] == labels[i]
		# print(relevant)

		if relevant.sum() == 0:  # Skip if there are no correct matches
			continue

		# print(sorted_indices[i].size())
		# Compute Precision@k for relevant positions
		# precisions = torch.cumsum(relevant, dim=0) / torch.arange(1, N, device=features.device).float()
		precisions = torch.cumsum(relevant, dim=0) / torch.arange(1, N, device=features.device).float()
		
		# Compute Average Precision (AP) for the current query
		ap = (precisions * relevant).sum() / relevant.sum()
		ap_list.append(ap)

	# Compute mean AP (mAP)
	mean_AP = torch.stack(ap_list).mean().item() if ap_list else 0.0

	return {'rank1': rank1_correct.item(), 'rank5': rank5_correct.item(), 'mAP': mean_AP}

class PetFaceDataset(Dataset):
    def __init__(self, image_path, class_names=None):
        self.class_names = class_names
        image_list = []
        for name in class_names:
            filenames = pd.read_csv(os.path.join(image_path, "split", name, "train.csv"))['filename']
            image_list.extend([os.path.join(image_path, 'images', f) for f in filenames])
        d = {}
        self.labels = []
        self.label2images = {}
        self.categorytoindex = {c: i for i, c in enumerate(class_names)}
        self.category_labels = []
        for idx, f in enumerate(image_list): # f:beverage_bottle/00018_00002/00000.jpg
            n = '/'.join(f.split('/')[-3:-1])  # n:beverage_bottle/00018_00002 ，格式是：类别/个体ID
            if n not in d:
                d[n] = len(d) # d给个体分配唯一ID，也就是pet
            self.labels.append(d[n])  # 每条记录的pet
            if n not in self.label2images:
                self.label2images[n] = [] # pet -> 个体的所有图片
            # self.label2images[n].append(idx)
            self.label2images[n].append(f)
            self.category_labels.append(self.categorytoindex[n.split('/')[0]])
        self.image_list = image_list
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True),
        ])

    def __len__(self):
        return len(self.label2images)
    
    def __getitem__(self, idx):
        pid = list(self.label2images.keys())[idx] # idx指向某个petid
        img_path1, img_path2 = random.choices(self.label2images[pid], k=2) # 从petid对应的图片中随机选两张
        image1 = self.transform(Image.open(img_path1).convert("RGB"))
        image2 = self.transform(Image.open(img_path2).convert("RGB"))
        return {'image_crops': torch.stack([image1, image2]), 'labels': torch.tensor(idx, dtype=torch.long)} # 每个样本是同一个pet的2张图

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_name: str = field(default=10)
    cluster_index: int = field(default=-1)
    vision_model: str = field(default='dinov2_vitb14')
    llm_model: str = field(default='Qwen/Qwen3-0.6B')
    num_id_tokens: int = field(default=32)
    num_vpt_tokens: int = field(default=32)
    num_icl_samples: int = field(default=64)
    num_icl_bs: int = field(default=1)
    icl_loss_weight: float = field(default=1.0)
    ot_loss_weight: float = field(default=0.01)


from custom_trainer import CustomTrainer
from config import data_config
class MyTrainTask(CustomTrainer):

    def __init__(self, args):
        weight_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

        '''
        data_config = {
            'amazon': {
                'splits': [
                    ['bicycle_helmet', 'hand_truck', 'jacket', 'duffel_bag', 'cooler', 'binoculars', 'bicycle'],
                    ['purse', 'skateboard', 'suitcase', 'tire_wheel', 'mobile_phone', 'hat', 'shoes'],
                    ['backpack', 'portable_speaker', 'stroller', 'beverage_bottle', 'food_container', 'box', 'cart'],
                    ['headphones', 'trash_can', 'poster_tube', 'pet_carrier', 'musical_instrument', 'book', 'tackle_box'],
                    ['sports_equipment', 'portable_chair', 'sports_ball', 'bucket', 'umbrella', 'hardshell_case']
                ],
                'classes': ['tackle_box', 'portable_chair', 'bicycle', 'poster_tube', 'duffel_bag', 'sports_equipment', 'hat', 'box', 'purse', 'hardshell_case', 'suitcase', 'umbrella', 'musical_instrument', 'trash_can', 'hand_truck', 'pet_carrier', 'sports_ball', 'mobile_phone', 'portable_speaker', 'binoculars', 'headphones', 'beverage_bottle', 'tire_wheel', 'jacket', 'cooler', 'book', 'stroller', 'backpack', 'food_container', 'cart', 'bucket', 'shoes', 'bicycle_helmet', 'skateboard'],
                'root': './groundingdino_cropped',
                'transform': None,
            }
        }
        '''
        if args.cluster_index == -1:
            train_categories = data_config[args.dataset_name]['classes']
        else:
            classes = []
            for i in range(len(data_config[args.dataset_name]['splits'])):
                if args.cluster_index == i:
                    continue
                classes.extend(data_config[args.dataset_name]['splits'][i])
            train_categories = classes
        self.categories = train_categories
        # 宠物类型 -> 不同的图片
        train_dataset = PetFaceDataset(
            image_path=data_config[args.dataset_name]['root'],
            class_names=train_categories,
        )

        print('num samples: ', len(train_dataset))

        from models import Model
        model = Model(args)
        model.to(dtype=weight_dtype).cuda()
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.to(dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        extra_losses = ['ot_loss', 'id_loss', 'icl_loss', 'std']
        super().__init__(model=model,
                         args=args,
                         extra_losses=extra_losses,
                        #  data_collator=DataCollatorForSupervisedDataset(tokenizer),
                         # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
                         optimizers=[optimizer, None],
                         train_dataset=train_dataset,
                         eval_dataset=train_dataset,
                         )

    @torch.no_grad()
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self.model.eval()

        test_categories = []
        for name in data_config[self.args.dataset_name]['classes']:
            if self.args.cluster_index != -1:
                if name in self.categories:
                    continue
            test_categories.append(name)
        all_results = {}
        for name in test_categories:
            all_features, all_labels = [], []
            from PIL import Image
            from torch.utils.data import Dataset, DataLoader
            import tqdm
            
            train_dataset = PetFaceDataset(image_path=data_config[self.args.dataset_name]['root'], class_names=[name])
            train_dataloader = DataLoader(train_dataset, batch_size=self.args.num_icl_samples, shuffle=False, num_workers=16)
            batch = next(iter(train_dataloader))
            batch = {k: v.cuda() for k, v in batch.items()}
            prompts = self.model(**batch)['prompts']
            # print(prompts.size())

            transform = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((336, 336), interpolation=Image.BICUBIC),
                torchvision.transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            from ops.dataset import PetFaceReIDEvalDataset
            dataset = PetFaceReIDEvalDataset(data_config[self.args.dataset_name]['root'], class_name=name, transform=transform)
            from torch.utils.data import Dataset, DataLoader
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
            all_features, all_labels = [], []
            import tqdm
            for images1, labels in tqdm.tqdm(dataloader):
                with torch.no_grad():
                    def ext(images):
                        e = self.model(images, prompts=prompts)['features'] + self.model(torch.flip(images, dims=(3,)), prompts=prompts)['features']
                        import torch.nn.functional as F
                        e = F.normalize(e, p=2, dim=1)
                        return e
                    e1 = ext(images1.cuda())
                    all_features.append(e1)
                    all_labels.append(labels)
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            results = compute_rank1_rank5_map(all_features, all_labels.cuda())
            for k in results:
                all_results[name + '_' + k] = results[k]
        import numpy as np
        mean_rank1 = np.mean([all_results[k] for k in all_results if k.endswith('_rank1')])
        mean_rank5 = np.mean([all_results[k] for k in all_results if k.endswith('_rank5')])
        mean_mAP = np.mean([all_results[k] for k in all_results if k.endswith('_mAP')])
        self.log({'mean_rank1': mean_rank1, 'mean_rank5': mean_rank5, 'mean_mAP': mean_mAP})
        return all_results

    def _get_train_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:
        dataset = self.train_dataset
        pids = np.array(list(dataset.label2images.keys()))
        labels = np.array([x.split('/')[0] for x in pids])
        label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        # num_batches = self.args.max_steps
        num_batches = 100000

        batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes
        print('batch_size: ', batch_size, 'num_batches: ', num_batches)
        all_batches = []
        
        while len(all_batches) < num_batches:
            current_label = np.random.choice(list(label_to_indices.keys()))
            indices = label_to_indices[current_label]
            batch_indices = np.random.choice(indices, batch_size, replace=True)
            all_batches.append(batch_indices)
        indices = np.array(all_batches).flatten()
        from torch.utils.data import DataLoader, Dataset, Sampler
        class OrderedSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices  # List of specific indices

            def __iter__(self):
                return iter(self.indices)  # Yield indices in order

            def __len__(self):
                return len(self.indices)
        return OrderedSampler(indices)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(TrainingArguments)

    training_args = parser.parse_args_into_dataclasses()[0]
    os.environ["WANDB_PROJECT"] = "ObjectReID"
    print(training_args)

    trainer = MyTrainTask(args=training_args)

    # trainer.train()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # trainer.save_state()
