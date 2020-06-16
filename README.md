### 该仓库为 KDD Cup 2020 Challenges for Modern E-Commerce Platform: Multimodalities Recall 第八名开源代码

**详细的技术报告见[technique_report.pdf](file:///technique_report.pdf)**

本仓库已将各个模型的预测概率上传[prob](file:///user_data/embeding_data)，可直接运行最后lgb集成部分。

# 运行说明

1. 切换到code目录下

2. ```bash
   pip install -r requirement.txt
   ```

3. ```bash main.sh``` ，最终预测的文件将会在```prediction_result```路径下生成


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