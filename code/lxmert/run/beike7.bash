# # The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=../../user_data/lxmert_model/$name
result_output=../../user_data/lxmert_model/result
mkdir -p $result_output/val
mkdir -p $result_output/test
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash



CUDA_VISIBLE_DEVICES=0,1 python src/tasks/vqa.py \
    --train train --valid valid --test testB \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA ../../external_resources/pretrained/model \
    --basedatadir '../../user_data/lxmert_processfile/' \
    --basebert '../../external_resources/pretrained/bert/bert-base-uncased.tar.gz' \
    --batchSize 200 --optim bert --lr 6e-5 --epochs 32 --max_area_boxes 5 --ROI_select 0 \
    --tqdm --output $output ${@:3}