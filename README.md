# 运行说明
**由于NN的训练受到多因素的影响，包括GPU数量及型号，随机数等，下面的可执行代码仅包含embedding部分代码，NN部分在/code下可以找到并运行**
1. 切换到code目录下
2. ```bash main.sh``` 运行embedding代码，最终预测的文件将会在```prediction_result```路径下生成


# 文件夹路径及说明
project
	|--README.md
	|--technique_report.pdf  
    |-- data
        |-- multimodal_labels.txt
        |-- train    
            |-- train.tsv               # 未放入
        |-- valid
            |-- valid.tsv
            |-- valid_answer.json
        |-- testA
            |-- testA.tsv
        |-- testB
            |-- testB.tsv
	|--external_resources
        |-- external_files.md           # 公共预训练模型列表及下载路径
	|--user_data
        |-- embeding_data               # embedding过程中使用到的概率文件
		|-- model_data
			|-- model.md                # 所有本文训练好的模型的百度网盘下载地址
        |-- lxmert_processfile          # lxmert预处理后的文件
	|--prediction_result                # 最终输出
	|--code
		|-- main.py or main.sh          # 调用 lgb_embedding.py 
        |-- lgb_embedding.py            # embedding 部分代码
        |-- LGB_lda.py                  
        |-- LGB_word2vec.py
        |-- local_val.py                # valid上的评分脚本
        |-- lxmert                      # lxmert 代码
        |-- lxmert-hard_sample          # hard_sample 部分lxmert代码