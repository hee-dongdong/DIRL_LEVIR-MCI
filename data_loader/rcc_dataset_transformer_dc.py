import os
import json
import numpy as np
import random
import time
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.preprocessing import StandardScaler
from PIL import Image


class RCCDataset(Dataset):
    shapes = set(['ball', 'block', 'cube', 'cylinder', 'sphere'])
    sphere = set(['ball', 'sphere'])
    cube = set(['block', 'cube'])
    cylinder = set(['cylinder'])

    colors = set(['red', 'cyan', 'brown', 'blue', 'purple', 'green', 'gray', 'yellow'])

    materials = set(['metallic', 'matte', 'rubber', 'shiny', 'metal'])
    rubber = set(['matte', 'rubber'])
    metal = set(['metal', 'metallic', 'shiny'])

    type_to_label = {
        'color': 0,
        'material': 1,
        'add': 2,
        'drop': 3,
        'move': 4,
        'no_change': 5
    }

    def __init__(self, cfg, split):
        self.cfg = cfg

        print('Speaker Dataset loading vocab json file: ', cfg.data.vocab_json)
        self.vocab_json = cfg.data.vocab_json
        self.word_to_idx = json.load(open(self.vocab_json, 'r'))
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.idx_to_word)
        print('vocab size is ', self.vocab_size)

        self.d_feat_dir = cfg.data.default_feature_dir
        self.s_feat_dir = cfg.data.semantic_feature_dir

        self.d_feats = sorted(os.listdir(self.d_feat_dir))
        self.s_feats = sorted(os.listdir(self.s_feat_dir))

        assert len(self.d_feats) == len(self.s_feats), \
            'The number of features are different from each other!'

        self.d_img_dir = cfg.data.default_img_dir
        self.s_img_dir = cfg.data.semantic_img_dir

        self.d_imgs = sorted(os.listdir(self.d_img_dir))
        self.s_imgs = sorted(os.listdir(self.s_img_dir))

        self.splits = json.load(open(cfg.data.splits_json, 'r'))
        self.split = split

        if split == 'train':
            self.batch_size = cfg.data.train.batch_size
            self.seq_per_img = cfg.data.train.seq_per_img
            self.split_idxs = self.splits['train']
            self.num_samples = len(self.split_idxs)
            if cfg.data.train.max_samples is not None:
                self.num_samples = min(cfg.data.train.max_samples, self.num_samples)
        elif split == 'val':
            self.batch_size = cfg.data.val.batch_size
            self.seq_per_img = cfg.data.val.seq_per_img
            self.split_idxs = self.splits['val']
            self.num_samples = len(self.split_idxs)
            if cfg.data.val.max_samples is not None:
                self.num_samples = min(cfg.data.val.max_samples, self.num_samples)
        elif split == 'test':
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = self.splits['test']
            self.num_samples = len(self.split_idxs)
            if cfg.data.test.max_samples is not None:
                self.num_samples = min(max_samples, self.num_samples)
        else:
            raise Exception('Unknown data split %s' % split)

        print("Dataset size for %s: %d" % (split, self.num_samples))

        flat_img_root = Path(self.d_img_dir).resolve()
        dataset_root = flat_img_root.parent.parent
        self.label_dir = dataset_root / 'images' / split / 'label_rgb'
        self.label_prefix = split
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        self.label_files = sorted(self.label_dir.glob('*.png'))
        if len(self.label_files) != 0 and len(self.label_files) != self.num_samples:
            raise ValueError(f"Number of auxiliary labels ({len(self.label_files)}) "
                             f"does not match split size ({self.num_samples}) for split '{split}'.")

        # load in the sequence data
        self.h5_label_file = h5py.File(cfg.data.h5_label_file, 'r')
        seq_size = self.h5_label_file['labels'].shape
        self.labels = self.h5_label_file['labels'][:]  # just gonna load...

        self.max_seq_length = seq_size[1]
        self.IGNORE = -1
        self.label_start_idx = self.h5_label_file['label_start_idx'][:]
        self.label_end_idx = self.h5_label_file['label_end_idx'][:]
        print('Max sequence length is %d' % self.max_seq_length)
        self.h5_label_file.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_idx = self.split_idxs[index]

        # Fetch image data
        # one easy way to augment data is to use nonsemantically changed
        # scene as the default :)
        if self.split == 'train':
            d_feat_path = os.path.join(self.d_feat_dir, self.d_feats[img_idx])
            d_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])

        else:
            d_feat_path = os.path.join(self.d_feat_dir, self.d_feats[img_idx])
            d_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])


        q_feat_path = os.path.join(self.s_feat_dir, self.s_feats[img_idx])
        q_img_path = os.path.join(self.s_img_dir, self.s_imgs[img_idx])

        d_feature = torch.FloatTensor(np.load(d_feat_path))
        q_feature = torch.FloatTensor(np.load(q_feat_path))

        # Fetch sequence labels
        ix1 = self.label_start_idx[img_idx]
        ix2 = self.label_end_idx[img_idx]
        n_cap = ix2 - ix1 + 1

        seq = np.zeros([self.seq_per_img, self.max_seq_length], dtype=int)
        if n_cap < self.seq_per_img:
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :self.max_seq_length] = \
                    self.labels[ixl, :self.max_seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            seq[:, :self.max_seq_length] = \
                self.labels[ixl: ixl + self.seq_per_img, :self.max_seq_length]

        # Generate masks
        mask = np.zeros_like(seq)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), seq)))
        for ix, row in enumerate(mask):
            row[:nonzeros[ix]] = 1
        if seq.size == self.max_seq_length:
            labels_with_ignore_tolish = [self.IGNORE if m == 0 else tid
                      for tid, m in zip(seq.squeeze(0).tolist(), mask.squeeze(0).tolist())][1:] + [self.IGNORE]
            labels_with_ignore = np.array(labels_with_ignore_tolish)
            labels_with_ignore = np.expand_dims(labels_with_ignore, 0)


        else:
            labels_with_ignore = np.zeros_like(seq)


        aux_label = None
        if self.label_files:
            label_path = self.label_files[index]
            # Load as RGB
            aux_label_pil = Image.open(label_path).convert('RGB')
            aux_label_np = np.array(aux_label_pil)
            
            # Create empty mask (Background=0)
            aux_label = torch.zeros((aux_label_np.shape[0], aux_label_np.shape[1]), dtype=torch.long)
            
            # Map colors to indices
            # (255, 0, 0) -> 1
            mask_1 = (aux_label_np[:, :, 0] == 255) & (aux_label_np[:, :, 1] == 0) & (aux_label_np[:, :, 2] == 0)
            aux_label[mask_1] = 1
            
            # (255, 255, 0) -> 2
            mask_2 = (aux_label_np[:, :, 0] == 255) & (aux_label_np[:, :, 1] == 255) & (aux_label_np[:, :, 2] == 0)
            aux_label[mask_2] = 2
            
            # Add more mappings if needed based on dataset specs


        return (d_feature, q_feature,
                seq, labels_with_ignore, mask, aux_label,
                d_img_path, q_img_path)

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length


def rcc_collate(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]

    q_feat_batch = transposed[1]
    seq_batch = default_collate(transposed[2])
    label_with_ignore_batch = default_collate(transposed[3])

    mask_batch = default_collate(transposed[4])
    aux_label_batch = default_collate(transposed[5])

    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)

    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    d_img_batch = transposed[6]

    q_img_batch = transposed[7]

    return (d_feat_batch, q_feat_batch,
            seq_batch, label_with_ignore_batch,
            mask_batch, aux_label_batch,

            d_img_batch, q_img_batch)


class RCCDataLoader(DataLoader):

    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = rcc_collate
        super().__init__(dataset, **kwargs)
