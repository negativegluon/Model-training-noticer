# Model-training-noticer
基于tensorboard日志与GPU使用情况检测模型训练有没有出现异常状况，如果有则通过napcat发送至自己的qq

## 配置与使用

在本地配置[napcat](https://github.com/NapNeko/NapCatQQ) 服务。  
打开一个http服务器，不要设置token。  
修改两个template文件，message_handler_config中的token部分不需要填写。然后将两个文件删除`_template`后缀。  
安装`tensorboard`与`pynvml`库。  

运行`main.py`文件即可。  

## 问题
现阶段没有查到怎么用request访问napcatHTTP服务器时添加token，之后会改进。  
只推荐在本机/校内网络中使用。
