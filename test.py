import os
import argparse
import json
import time
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from configs.config_transformer import cfg, merge_cfg_from_file
from data_loader.datasets import create_dataset
from models.DIRL import DIRL, AddSpatialInfo
from models.new_CCR import CCR

from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, decode_sequence_transformer, coco_gen_format_save
from utils.vis_utils import visualize_att
from tqdm import tqdm

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--save_masks', action='store_true', help='Save auxiliary predicted masks')
parser.add_argument('--num_mask_samples', type=int, default=50, help='Number of mask samples to save')
parser.add_argument('--snapshot', type=int, required=True)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
# assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

# Device configuration
use_cuda = torch.cuda.is_available()
if args.gpu == -1:
    gpu_ids = cfg.gpu_id
else:
    gpu_ids = [args.gpu]
torch.backends.cudnn.enabled  = True
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name

output_dir = os.path.join(exp_dir, exp_name)

test_output_dir = os.path.join(output_dir, f'test_output_{args.snapshot}')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)
caption_output_path = os.path.join(test_output_dir, 'captions', 'test')
if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)
att_output_path = os.path.join(test_output_dir, 'attentions', 'test')
if not os.path.exists(att_output_path):
    os.makedirs(att_output_path)

# Mask output directory
mask_output_path = os.path.join(test_output_dir, 'predicted_masks')
if args.save_masks and not os.path.exists(mask_output_path):
    os.makedirs(mask_output_path)

if args.visualize:
    visualize_save_dir = os.path.join(test_output_dir, 'visualizations')
    if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

snapshot_dir = os.path.join(output_dir, 'snapshots')
snapshot_file = '%s_checkpoint_%d.pt' % (exp_name, args.snapshot)
snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
checkpoint = load_checkpoint(snapshot_full_path)
change_detector_state = checkpoint['change_detector_state']
speaker_state = checkpoint['speaker_state']


# Load modules
change_detector = DIRL(cfg)
change_detector.load_state_dict(change_detector_state)
change_detector = change_detector.to(device)

speaker = CCR(cfg)
speaker.load_state_dict(speaker_state)
speaker.to(device)

spatial_info = AddSpatialInfo()
spatial_info.to(device)

print(change_detector)
print(speaker)
print(spatial_info)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
idx_to_word = train_dataset.get_idx_to_word()
test_dataset, test_loader = create_dataset(cfg, 'test')

# Color map for mask visualization (3 classes)
# Class 0: Background, Class 1: Red (255,0,0), Class 2: Yellow (255,255,0)
MASK_COLORS = [
    [0, 0, 0],        # 0: Background - Black
    [255, 0, 0],      # 1: Red
    [255, 255, 0],    # 2: Yellow
]

def save_mask_visualization(pred_mask, gt_mask, d_img_path, save_path, image_id):
    """
    Save predicted mask and ground truth mask as visualization.
    
    Args:
        pred_mask: predicted mask (H, W) with class indices
        gt_mask: ground truth mask (H, W) with class indices or None
        d_img_path: path to the original image
        save_path: directory to save the visualization
        image_id: identifier for the image
    """
    pred_mask_np = pred_mask.cpu().numpy()
    
    # Create colored prediction mask
    h, w = pred_mask_np.shape
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(MASK_COLORS):
        pred_colored[pred_mask_np == cls_idx] = color
    
    # Create figure
    if gt_mask is not None:
        gt_mask_np = gt_mask.cpu().numpy()
        gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, color in enumerate(MASK_COLORS):
            gt_colored[gt_mask_np == cls_idx] = color
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Load and show original image
        try:
            orig_img = Image.open(d_img_path).resize((256, 256))
            axes[0].imshow(orig_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
        except:
            axes[0].set_title('Original Image (Not Found)')
            axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        axes[2].imshow(gt_colored)
        axes[2].set_title('Ground Truth Mask')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Load and show original image
        try:
            orig_img = Image.open(d_img_path).resize((256, 256))
            axes[0].imshow(orig_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
        except:
            axes[0].set_title('Original Image (Not Found)')
            axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{image_id}_mask.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Also save just the predicted mask as a separate image
    pred_img = Image.fromarray(pred_colored)
    pred_img.save(os.path.join(save_path, f'{image_id}_pred_only.png'))


set_mode('eval', [change_detector, speaker])
mask_sample_count = 0

with torch.no_grad():
    test_iter_start_time = time.time()

    result_sents_pos = {}
    result_sents_neg = {}
    for i, batch in tqdm(enumerate(test_loader)):

        d_feats, sc_feats, \
        labels, labels_with_ignore, masks, aux_labels, \
        d_img_paths, sc_img_paths = batch

        val_batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)

        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(
            device)

        diff_bef_pos, diff_aft_pos, dirl_loss = change_detector(d_feats, sc_feats)

        speaker_output_pos, pos_att = speaker.sample(diff_bef_pos, diff_aft_pos, sample_max=1)

        gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:])

        # Save predicted masks if requested
        if args.save_masks and mask_sample_count < args.num_mask_samples:
            aux_pred, aux_pred_class = speaker.get_auxiliary_mask(diff_bef_pos, diff_aft_pos)
            
            for j in range(val_batch_size):
                if mask_sample_count >= args.num_mask_samples:
                    break
                    
                image_id = d_img_paths[j].split('/')[-1].split('.')[0]
                
                # Get ground truth mask if available
                gt_mask = aux_labels[j] if aux_labels is not None else None
                
                save_mask_visualization(
                    aux_pred_class[j],
                    gt_mask,
                    d_img_paths[j],
                    mask_output_path,
                    image_id
                )
                mask_sample_count += 1
                
                if mask_sample_count % 10 == 0:
                    print(f'Saved {mask_sample_count}/{args.num_mask_samples} mask samples')

        for j in range(val_batch_size):
            gts = decode_sequence_transformer(idx_to_word, labels[j][:, 1:])

            sent_pos = gen_sents_pos[j]

            image_id = d_img_paths[j].split('/')[-1]
            result_sents_pos[image_id] = sent_pos

            image_num = image_id.split('.')[0]


    test_iter_end_time = time.time() - test_iter_start_time
    print('Test took %.4f seconds' % test_iter_end_time)

    if args.save_masks:
        print(f'Saved {mask_sample_count} mask visualizations to {mask_output_path}')

    result_save_path_pos = os.path.join(caption_output_path, 'sc_results.json')

    coco_gen_format_save(result_sents_pos, result_save_path_pos)
