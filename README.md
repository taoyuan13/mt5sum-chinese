## 数据集

* 中文数据集：**NLPCC2016** (Chinese Long Text Summarization Dataset)




## 预处理

###Step 1 下载原始数据

将下载好的NLPCC 2017原始数据集文件放入`BertSum-master_Chinese/raw_data`

###Step 2 将原始文件转换成json文件存储和划分数据集

`BertSum-master_Chinese/src`目录下，运行：

```
python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data
```

###Step 3 分句分词 & 分割文件 & 进一步简化格式

* 分句分词：首先按照符号['。', '！', '？']分句，若得到的句数少于2句，则用['，', '；']进一步分句

* 分割文件：训练集文件太大，分割成小文件便于后期训练。**分割后，每个文件包含不多于16000条记录**

`BertSum-master_Chinese/src`目录下，运行：

```
python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data
```

###Step 4 句子标注 & 训练前预处理

* 句子预处理：找出与参考摘要最接近的n句话(相似程度以ROUGE衡量)，标注为1(属于摘要)

```
python preprocess_LAI.py -mode format_to_mt5 -encoder mt5 -raw_path ../json_data -save_path ../mt5_data -oracle_mode greedy -n_cpus 2 -max_position_embeddings 1024
```



## 模型训练


### 抽取式设置

#### Mt5Ext
```
python train_LAI.py -task ext -mode train -model mt5 -extractor cls -corpora NLPCC -visible_gpus 0 -batch_size 1000 -train_steps 100000
```
-load_from_abstractive


### 生成式设置

#### TransformerAbs (baseline)
```
python train_LAI.py -task abs -mode train -model baseline -corpora NLPCC -dec_dropout 0.1 -lr 0.05 -warmup_steps 8000 -save_checkpoint_steps 2000 -batch_size 64 -accum_count 5 -train_steps 300000 -visible_gpus 0
```
#### BertAbs
```
python train_LAI.py -task abs -mode train -model bert -corpora NLPCC -sep_optim true -save_checkpoint_steps 2000 -batch_size 64 -accum_count 5 -train_steps 300000 -visible_gpus 0
```
#### Mt5Abs
```
python train_LAI.py -task abs -mode train -model mt5 -corpora NLPCC -save_checkpoint_steps 2000 -batch_size 64 -accum_count 5 -train_steps 300000 -visible_gpus 0 -copy true -load_from_extractive model_step_100000.pt
```


### 多任务设置

#### Mt5Mtl
```
python train_LAI.py -task mtl -mode train -model mt5 -extractor cls -corpora NLPCC -batch_size 64 -accum_count 5 -train_steps 100000 -visible_gpus 0 -copy true -mtl_loss_mode covweighting -load_from_extractive model_step_100000.pt -load_from_abstractive model_step_96000.pt
```

**提醒**：如果训练过程被意外中断，可以通过 -train_from 从某个节点继续训练(-save_checkpoint_steps设置了定期储存模型信息)





## 模型评估

模型训练完毕后，`BertSum-master_Chinese/src`目录下，运行：

-mode validate 将不断地检查模型所在目录并为每个新保存的检查点评估模型，和训练一起使用（实时的）
#### Mt5Ext
```
python train_LAI.py -task ext -mode validate -model mt5 -extractor cls -corpora NLPCC -visible_gpus 0
```
#### Mt5Abs
```
python train_LAI.py -task abs -mode validate -model mt5 -corpora NLPCC -visible_gpus 0 -copy true -batch_size 1 -test_batch_size 1
```
#### Mt5Mtl(multi-task learning)
```
python train_LAI.py -task mtl -mode validate -model mt5 -extractor cls -corpora NLPCC -visible_gpus 0 -copy true -mtl_loss_mode covweighting -batch_size 1 -test_batch_size 1
```

-mode test 需要使用 -test_from 来指定想要使用的检查点
```
python train_LAI.py -task ext -mode test -model mt5 -extractor cls -corpora LCSTS -visible_gpus 0 -test_from model_step_50000.pt
```
-mode validate 和 -test_all 系统将加载所有保存的检查点并选择最好的来生成摘要（耗时较长）
```
python train_LAI.py -task ext -mode validate -model mt5 -extractor cls -corpora LCSTS -visible_gpus 0 -test_all
```


## 生成Cluster聚类摘要
```
python train_LAI.py -mode cluster -model mt5 -corpora NLPCC -visible_gpus 0 -log_file ../logs/cluster_nlpcc.log -result_path ../results/cluster_nlpcc -test_from model_step_50000.pt
```

## 生成Oracle/Lead摘要

Oracle摘要：使用贪婪算法，在原文中找到与参考摘要最相近n句话(原代码设置n=3，可自行调整)
Lead摘要：在原文中找到与Oracle摘要数量相同的原文中最前面的n句话

-mode lead或者oracle
```
python train_LAI.py -mode oracle -corpora NLPCC -visible_gpus 0 -log_file ../logs/oracle_nlpcc.log -result_path ../results/oracle_nlpcc
```
摘要大小调整方法：
目录`BertSum-master_Chinese/src/prepro/`：
data_builder_LAI.py: line204 - oracle_ids = greedy_selection(source, tgt, **3**)


## 新数据集训练

如果要在新数据集上使用BERTSUM，只需：

* 原始数据格式整理成`BertSum-master_Chinese/raw_data/NLPCC_test.json`文件中数据对应格式
* 相应文件名／路径名也要做调整如：`-final_data_path ../bert_data/NLPCC` `-log_file NLPCC_oracle` (NLPCC改成对应名称)
* 调整完后，预处理部分从**Step 3**开始即可