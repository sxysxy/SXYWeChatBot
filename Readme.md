# SXYWeChatBot

一个接入了ChatGPT和NovelAI的微信聊天机器人，兼容windows、mac、linux平台，代码很短很简单很容易扩展别的功能～ 

## 安装配置方法 (#ch1)
### 依赖 (#ch11)
### 修改配置 (#ch12)
### 然后就可以运行了 (#ch13)
### 注意 (#ch14)

## 安装配置方法
<p id="ch1"> </p>

### 依赖
<p id="ch11> </p>

用到了两种编程语言：go和python3。使用go是因为本项目依赖于强力的使用go写成的<a href="https://github.com/eatmoreapple/openwechat">openwechat</a>实现对微信会话的获取以及发送消息的功能。调用ChatGPT以及Stable Diffusion模型则使用python3。

python3需要再安装这些库，使用pip安装就可以：

```
torch flask openai diffusers 
```

```
pip install torch flask openai diffusers
```

当然如果使用cuda加速建议按照<a href="https://pytorch.org">pytorch官网</a>提供的方法安装支持cuda加速的torch版本。

Apple Silicon的macbook上可以使用mps后端加速，我开发的时候使用的就是M1 Max芯片的Macbook Pro。

### 修改配置
<p id="ch12"> </p>

你需要有一个OpenAI账号，然后将API Key写到config.json的OpenAI-API-Key字段后，然后保存。

其余的配置通常按照默认的就可以，或者可以前往OpenAI官网查看其他可用的GPT模型，或者到huggingface上查看其他可用的Stable Diffusion模型。

### 然后就可以运行了
<p id="ch13> </p>


Mac/Linux用户可以直接运行start.sh：

```
./start.sh config.json
```

或者分开运行bot和wechat_client：

```
go run wechat_client.go
python bot.py
```

### 注意
<p id="ch14"> </p>

第一次运行需要下载Stable Diffusion模型，默认的stabilityai/stable-diffusion-2-1有将近10GB，并且从外网下载，需要有比较快速稳定的网络条件。

## 机器人使用方法
<p id="ch2"> </p>

