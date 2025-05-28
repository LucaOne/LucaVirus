# 训练脚本       
## 从头开始训练         
基于LucaOne的chenckpoint作为模型的初始化             
脚本：**run_multi_v1.0.sh**        


## 中途发生失败，继续训练     
很难保证大规模训练中间不失败，从而需要从最近的失败点继续训练      
那么就是：加载LucaVirus的已经保存的checkpoint           
脚本：**run_multi_v1.0_continue.sh**      


## 只使用两个MASK任务做训练      
不使用其他的预训练任务     
脚本：**run_multi_mask_v1.0.sh**        


## 只使用Gene数据做训练       
脚本：**run_multi_v1.0_gene.sh**          


## 只使用Prot数据做训练       
脚本：**run_multi_v1.0_prot.sh**        


## 单卡训练   
脚本：**run_multi_v1.0_single.sh**


# 训练超参数         
这里使用脚本`run_multi_v1.0.sh`来说明，一些重要的参数，其他使用默认值                

* MAX_LENGTH        
最长序列长度，包括首尾两个特殊字符[CLS]与[SEP]，比如3074 或者4098，根据模型大小与显存大小        
模型大小、batch_size、MAX_LENGTH三者共分显卡，也就是三者取值不可兼大            


* NUM_ATTENTION_LAYERS         
Transformer层数，最长序列长度MAX_LENGTH大一点，那么NUM_ATTENTION_LAYERS就需要小一点，默认12       


* NUM_ATTENTION_HEADS        
Transformer头数，需要被embedding_dim=2560整除，默认20         


* gradient_accumulation_steps     
梯度累积，正常情况训练是min batch训练，也就是一次性batch_size=k个样本同时计算，loss与梯度取平均值，但是由于显卡大小的问题，需要使用梯度累积来达到batch_size=k的效果，         
设置gradient_accumulation_steps=k，一般建议是k=16，32，或者64        


* batch_size=1       
大模型训练为了让模型更大与处理更长序列，使用batch_size=1 加上gradient_accumulation_steps=k来达到一样的效果         


* loss_logging_steps         
间隔多少个step在log文件中写入loss（实际上是gradient_accumulation_steps与loss_logging_steps的最小公倍数)         


* logging_steps      
间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数)    


* save_steps   
checkpoint的间隔step数保存一次(训练了n_gpu * save_steps个样本保存一个checkpoint)
**根据数据量设置**   


* warmup_steps        
warmup_steps个step到达peak lr，实际上是warmup_steps=warmup_steps/gradient_accumulation_steps    
一般是1000 * gradient_accumulation_steps     


* max_steps  
最大迭代step次数(这么多次后，peak lr变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）     
最后想要变成多大的值比如从lr1->lr2，那么就是(max_epochs*sample_cnt)*lr1/(n_gpu * batch_size * gradient_accumulation_steps*lr2))进行估算         
比如：15718611 * 4 * 2e-4/(8 * 1 * 32 * 5e-5)=982413        
**根据数据量设置**   


* learning_rate        
peak lr， 一般设置为e-4级别    


* worker_num      
数据加载器worker数        
**根据显卡数量设置**  


* CUDA_VISIBLE_DEVICES       
使用的显卡集合       
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    
**根据显卡数量设置**        


* model_dirpath       
使用已经训练好的模型进行参数初始化的模型的checkpoint所在的文件夹路径         


* nproc_per_node       
gpu卡数        
**根据显卡数量设置**       


## 对于中间失败而继续训练   
这种情况，模型在训练中会定期保存已经遍历过的样本的sample_id，继续跑则会跳过这些样本         


另外设置参数：      
* trained_checkpoint      
中断点最近保存的checkpoint     


* model_dirpath       
中断点最近保存的checkpoint所在的文件夹路径         


