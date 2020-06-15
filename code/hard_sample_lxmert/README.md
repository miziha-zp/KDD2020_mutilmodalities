# 说明
embedding中包含hard_sample的四个模型，来源为
   1. 使用lxmert预训练模型（我们完成的），1/3概率使用hard_sample作为负样本，1/3概率使用随机选取当前batch内的query作为负样本 9epoch，lr=6e-5 [n_query_BEST.pth]
   2. step1 再训练3epoch    [n_query_0608_newBEST.pth]
   3. step1的预训练模型；4/5概率使用hard_sample作为负样本，1/5概率使用随机选取当前batch内的query作为负样本 9epoch，lr=6e-5   [most_hard_0610_new_hard_BEST.pth]
   4. step1的预训练模型；4/5概率使用hard_sample作为负样本，1/5概率使用随机选取当前batch内的query作为负样本 9epoch，lr=6e-10   [most_hard_0609hard_BEST.pth]

# 运行说明
1. 运行 query_negative_sample.py 制作 hard_sample
2. 运行 对应的shell文件（shell文件名对应“说明”中顺序）
3. 四个完成训练的模型都可以在【链接：https://pan.baidu.com/s/1gi8IZJMZhgbqcKFMN2V9Og 提取码：9ui4】中找到