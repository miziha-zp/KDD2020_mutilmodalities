result_output=../../user_data/lxmert_model/result
mkdir -p $result_output/val
mkdir -p $result_output/test

CUDA_VISIBLE_DEVICES=0
    python src/tasks/get_prob.py \
    --test testB  --valid valid \
    --basedatadir '../../user_data/lxmert_processfile/' \
    --basebert '../../external_resources/pretrained/bert/bert-base-uncased.tar.gz' \
    --load '../../user_data/lxmert_model/beike1/beike1.pth' \
    --name 'beike1' \

    
