# LucaVirus     
LucaVirus: Modeling the Evolutionary and Functional Landscape of Viruses with a Unified Genome-Protein Language Model    
# TimeLine   

## 1. LucaVirus Workflow

## 2. LucaVirus PreTraining Data & PreTraining Tasks

## 3. Downstream Tasks

## 4. Environment Installation
### step1: update git
#### 1) centos
sudo yum update     
sudo yum install git-all

#### 2) ubuntu
sudo apt-get update     
sudo apt install git-all

### step2: install python 3.9
#### 1) download anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

#### 2) install conda
sh Anaconda3-2022.05-Linux-x86_64.sh
##### Notice: Select Yes to update ~/.bashrc
source ~/.bashrc

#### 3) create a virtual environment: python=3.9.13
conda create -n lucavirus python=3.9.13


#### 4) activate lucavirus
conda activate lucavirus  

##### create kernel   
conda create -n lucavirus python=3.9
conda activate lucavirus
conda install ipykernel
python -m ipykernel install --user --name lucavirus --display-name "Python(LucaVirus)"

##### install kernels for Jupyter   
jupyter kernelspec list    

##### remove kernel      
jupyter kernelspec uninstall lucavirus    

### step3:  install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple   

## 5. Embedding           
using `src/get_embedding.py` or `src/embedding/get_embedding.py`      
usage information refer to `src/embedding/README.md`               

## 6. For Downstream Tasks    

## 7. Dataset       

## 8. Training Scripts    
**run_multi_v1.0.sh**     
use the LucaOne's chenckpoint(`step=17600000` or `36000000`) for LucaVirus training.
 
**run_multi_v1.0_continue.sh**     
continue training when an interruption occurs.        

**run_multi_mask_v1.0.sh**     
training LucaVirus only using mask pretrain task.   

**run_multi_v1.0_gene.sh**   
training LucaVirus only using viral gene(DNA + RNA) data.  

**run_multi_v1.0_prot.sh**   
training LucaVirus only using viral protein data.     

**run_multi_v1.0_single.sh**   
training LucaVirus only using one GPU card.    

### TensorBoard for Loss Curve      
tensorboard --logdir tb-logs --bind_all --port 8008    
 

## 9. Data and Code Availability         

## 10. Contributor             
<a href="https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en" title="Yong He">Yong He</a>,    
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&pli=1&user=Zhlg9QkAAAAJ" title="Yuan-Fei Pan">Yuan-Fei Pan</a>,      
<a href="https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en" title="Zhaorong Li">Zhaorong Li</a>,    
<a href="https://scholar.google.com/citations?user=1KJOH7YAAAAJ&hl=zh-CN&oi=ao" title="Mang Shi">Mang Shi</a>,    
Yuqi Liu

## 11. Citation              








