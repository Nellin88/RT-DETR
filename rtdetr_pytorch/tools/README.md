

Train/test script examples
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only` 


Tuning script examples
- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth` 


Export script examples
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


GPU do not release memory
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


Save all logs
- Appending `&> train.log 2>&1 &` or `&> train.log 2>&1`


Merge two flower datasets (COCO)
- All labels are remapped into one class by default: `flower`.
- `python tools/merge_coco_datasets.py --output-root dataset/flower_detection_merged`
- `python tools/merge_coco_datasets.py --output-root dataset/flower_detection_merged --valid-to test`
- `python tools/merge_coco_datasets.py --output-root dataset/flower_detection_merged --test-to-val-ratio 0.5`
- `python tools/merge_coco_datasets.py --output-root dataset/flower_detection_merged --dry-run`
- `python tools/merge_coco_datasets.py --output-root dataset/flower_detection_merged --overwrite --mode hardlink`
- `python tools/merge_coco_datasets.py --output-root dataset/flower_detection_merged --class-name flower`

