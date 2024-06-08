### 第1步：配置环境
```
pip install -r requirement_env_rl1.txt
```

### 第2步：设置工作目录
```
按照/utils/get_exp_data_path.py中设置一下工作目录，并把Simple_Tag.zip数据放进去
```

### 第3步：运行demo
```
export SUPPRESS_MA_PROMPT=1
cd MBAM
sh ./exp/simple_tag/_train0.sh
``` 

