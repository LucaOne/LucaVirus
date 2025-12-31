# 推送到 PIP    
## 1. 安装必要工具    
pip install --upgrade build twine    

## 2. 构建分发包    
在项目根目录下（pyproject.toml 所在位置）运行：  
```bash
# 这会生成 dist/ 文件夹，里面包含 .tar.gz 和 .whl 文件
python -m build
```  

## 3. 上传到 PyPI    
你需要先在 PyPI 官网 注册账号，并创建一个 API Token     
```bash
# 上传所有生成的包
python -m twine upload dist/*
```

# 推送到 Hugging Face    

## 1. 安装并登录    
pip install huggingface_hub    
huggingface-cli login   

## 2. 创建仓库     
huggingface-cli repo create lucavirus --type model    
from huggingface_hub import HfApi

api = HfApi()

# 这里的 path_or_fileobj 指向你之前 convert_weights.py 保存的目录
# repo_id 是你的 "用户名/仓库名"
api.upload_folder(
    folder_path="./checkpoints/LucaGroup/LucaVirus-default-step3.8M",
    repo_id="LucaGroup/LucaVirus-default-step3.8M",
    repo_type="model",
)

api.upload_folder(
    folder_path="./checkpoints/LucaGroup/LucaVirus-gene-step3.8M",
    repo_id="LucaGroup/LucaVirus-gene-step3.8M",
    repo_type="model",
)

api.upload_folder(
    folder_path="./checkpoints/LucaGroup/LucaVirus-prot-step3.8M",
    repo_id="LucaGroup/LucaVirus-prot-step3.8M",
    repo_type="model",
)
api.upload_folder(
    folder_path="./checkpoints/LucaGroup/LucaVirus-mask-step3.8M",
    repo_id="LucaGroup/LucaVirus-mask-step3.8M",
    repo_type="model",
)
api.upload_folder(
    folder_path="./checkpoints/LucaGroup/LucaVirus-mask-step1.4M",
    repo_id="LucaGroup/LucaVirus-mask-step1.4M",
    repo_type="model",
)
