"""
Evaluation script for RT-DETR
Computes COCO metrics on the validation set
"""

import torch
import torch.nn as nn 
import os 
import sys 
import gc
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.core.yaml_utils import create, merge_config
from src.data import get_coco_api_from_dataset
from src.solver.det_engine import evaluate


def main(args):
    """Main evaluation function"""
    # Normalize device string: 'gpu' -> 'cuda'
    if args.device.lower() == 'gpu':
        args.device = 'cuda'
    
    device = torch.device(args.device)
    
    print(f"Loading config from {args.config}")
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if not args.resume:
        raise AttributeError('Please provide a checkpoint via --resume/-r')
    
    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu') 
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    # Load model state
    cfg.model.load_state_dict(state)
    del checkpoint
    del state
    gc.collect()

    # Create wrapper model - just deploy the model, we'll handle postprocessing in evaluate
    model = cfg.model.deploy().to(device)
    model.eval()
    
    eval_split = args.split
    if eval_split == 'auto':
        eval_split = 'test' if 'test_dataloader' in cfg.yaml_cfg else 'val'

    if eval_split not in ('val', 'test'):
        raise ValueError(f"Unsupported split: {eval_split}")

    dataloader_key = f'{eval_split}_dataloader'
    if dataloader_key not in cfg.yaml_cfg:
        raise AttributeError(f"{dataloader_key} is missing from the config")

    # Setup data loader
    print(f"Setting up {eval_split} dataloader...")
    merge_config(cfg.yaml_cfg)
    eval_dataloader = create(dataloader_key)
    eval_dataloader.shuffle = cfg.yaml_cfg[dataloader_key].get('shuffle', False)
    eval_dataloader = dist.warp_loader(eval_dataloader, shuffle=eval_dataloader.shuffle)
    
    # Get COCO API for evaluation
    base_ds = get_coco_api_from_dataset(eval_dataloader.dataset)
    
    print(f"Running evaluation on {eval_split} set...")
    print(f"Device: {args.device}")
    print(f"Dataset size: {len(eval_dataloader.dataset)}")
    print("-" * 80)
    
    # Run evaluation
    with torch.no_grad():
        stats, coco_evaluator = evaluate(
            model, cfg.criterion, cfg.postprocessor, eval_dataloader, base_ds, device, cfg.output_dir
        )
    
    print("-" * 80)
    print("\n=== EVALUATION RESULTS ===\n")
    
    # Print detailed metrics
    if 'coco_eval_bbox' in stats:
        bbox_stats = stats['coco_eval_bbox']
        print("COCO Bbox Metrics:")
        print(f"  AP (IoU=0.50:0.95):           {bbox_stats[0]:.3f}")
        print(f"  AP (IoU=0.50):                {bbox_stats[1]:.3f}")
        print(f"  AP (IoU=0.75):                {bbox_stats[2]:.3f}")
        print(f"  AP (small objects):           {bbox_stats[3]:.3f}")
        print(f"  AP (medium objects):          {bbox_stats[4]:.3f}")
        print(f"  AP (large objects):           {bbox_stats[5]:.3f}")
        print(f"  AR (max detections per img=1): {bbox_stats[6]:.3f}")
        print(f"  AR (max detections per img=10): {bbox_stats[7]:.3f}")
        print(f"  AR (max detections per img=100): {bbox_stats[8]:.3f}")
        print(f"  AR (small objects):           {bbox_stats[9]:.3f}")
        print(f"  AR (medium objects):          {bbox_stats[10]:.3f}")
        print(f"  AR (large objects):           {bbox_stats[11]:.3f}")
    
    if 'coco_eval_masks' in stats:
        mask_stats = stats['coco_eval_masks']
        print("\nCOCO Segmentation Metrics:")
        print(f"  AP (IoU=0.50:0.95):           {mask_stats[0]:.3f}")
        print(f"  AP (IoU=0.50):                {mask_stats[1]:.3f}")
        print(f"  AP (IoU=0.75):                {mask_stats[2]:.3f}")
        print(f"  AP (small objects):           {mask_stats[3]:.3f}")
        print(f"  AP (medium objects):          {mask_stats[4]:.3f}")
        print(f"  AP (large objects):           {mask_stats[5]:.3f}")
        print(f"  AR (max detections per img=1): {mask_stats[6]:.3f}")
        print(f"  AR (max detections per img=10): {mask_stats[7]:.3f}")
        print(f"  AR (max detections per img=100): {mask_stats[8]:.3f}")
        print(f"  AR (small objects):           {mask_stats[9]:.3f}")
        print(f"  AR (medium objects):          {mask_stats[10]:.3f}")
        print(f"  AR (large objects):           {mask_stats[11]:.3f}")
    
    # Optionally save results to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RT-DETR on validation set')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (cpu/gpu/cuda)')
    parser.add_argument('-o', '--output', type=str, default='', help='Optional path to save results as JSON')
    parser.add_argument('--split', type=str, default='auto', choices=['auto', 'val', 'test'], help='Dataset split to evaluate')
    args = parser.parse_args()
    
    main(args)
