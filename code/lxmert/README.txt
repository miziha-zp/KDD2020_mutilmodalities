
注：我们使用的所有由lxmert模型得到的各query id 下所有product文件：val.txt和test.txt，均在训练阶段生成，
每训练一轮都会保存该epoch下的模型以及该模型产生的val.txt以及test.txt。同时也提供测试接口，读取训练完成
的模型，产生val.txt和test.txt。

关于lxmert模型的训练和测试脚本bash路径修改的说明：

首先执行prepare文件： python prepare.py

一. 训练阶段

1.1 切换到lxmert主目录下，在终端执行 bash run/train.bash 0 lxr_model
其中：
lxr_model为保存当次训练结果（模型，日志等）的文件夹名称，可修改为其他名称，并由程序自动生成该文件夹。

1.2 修改主目录下eval.py中，131行的standard_path路径为实际valid_answer.json路径。

1.3 修改run/train.bash中的路径：
其中：
--loadLXMERTQA   为 pretrain_path/model  pretrain_path为预训练模型model_LXRT.pth的上级目录，注意该路径没有引号
--basedatadir   为数据目录，包括了处理后的train.tsv, valid.tsv 和testB.tsv
--basebert    为bert-base-uncased.tar.gz的路径
--RESUME 和--path_checkpoint    为续训功能，--RESUME缺省为0，当值为1的时候启动续训，此时设置--path_checkpoint为
user_data/lxmert_model/checkpoint/文件夹下lxr_best.pth的路径


1.4 训练
beike1~beike12分别来自于以下的训练设定（需要修改的bash参数）：


beike1    lr = 6e-5  max_box =5 训练16轮
beike2    lr = 6e-5  max_box =5 训练19轮
beike3    lr = 5e-5  max_box =5 训练15轮
beike4    lr=6e-5    max_box = 20 训练24轮
beike5    lr=6e-5    max_box =5   训练30轮
beike6    lr=6e-5   max_box = 20 训练30轮
beike7    lr=6e-5    max_box =5   训练32轮
beike8    lr=6e-5    max_box =5   训练33轮
beike9   lr=6e-5    max_box =5   训练34轮

beike10是finetune自25轮max_box =5后启用了ROI Select机制 的第31轮
beike11是finetune自25轮max_box =5后启用了ROI Select机制 的第32轮
beike12是finetune自25轮max_box =5后启用了ROI Select机制 的第33轮

训练命令：
beike1:  执行命令：bash run/beike1.bash 0 beike1
beike2:  执行命令：bash run/beike2.bash 0 beike2
beike3:  执行命令：bash run/beike3.bash 0 beike3
beike4:  执行命令：bash run/beike4.bash 0 beike4
beike5:  执行命令：bash run/beike5.bash 0 beike5
beike6:  执行命令：bash run/beike6.bash 0 beike6
beike7:  执行命令：bash run/beike7.bash 0 beike7
beike8:  执行命令：bash run/beike8.bash 0 beike8
beike9:  执行命令：bash run/beike9.bash 0 beike9
beike10:  执行命令：1. bash run/beike101112_finetune.bash 0 beike10
                               2.bash run/beike10.bash 0 beike10
beike11:  执行命令：1. bash run/beike101112_finetune.bash 0 beike11
                               2.bash run/beike11.bash 0 beike11
beike12:  执行命令：1. bash run/beike101112_finetune.bash 0 beike12
                               2.bash run/beike12.bash 0 beike1


二. 测试阶段
实际上在训练的时候，每一轮都会计算出对应的test.txt和val.txt，这里只不过提供一个独立出来的测试接口。
比如测试beike1.pth的时候，只需要在lxmert目录下执行 bash run/test.bash

其中，test.bash如下：
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

    
如果要测试beike2.pth,只需修改对应的名称即可： bash run/test.bash

这时，test.bash如下：

result_output=../../user_data/lxmert_model/result
mkdir -p $result_output/val
mkdir -p $result_output/test

CUDA_VISIBLE_DEVICES=0
    python src/tasks/get_prob.py \
    --test testB  --valid valid \
    --basedatadir '../../user_data/lxmert_processfile/' \
    --basebert '../../external_resources/pretrained/bert/bert-base-uncased.tar.gz' \
    --load '../../user_data/lxmert_model/beike2/beike2.pth' \
    --name 'beike2' \




