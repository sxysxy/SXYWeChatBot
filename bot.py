import json
import openai
import re
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import flask
import typing
import traceback

ps = argparse.ArgumentParser()
ps.add_argument("--config", default="config.json", help="Configuration file")
args = ps.parse_args()

with open(args.config) as f:
    config_json = json.load(f)

class GlobalData:
   # OPENAI_ORGID = config_json[""]
    OPENAI_APIKEY = config_json["OpenAI-GPT"]["OpenAI-Key"]
    OPENAI_MODEL = config_json["OpenAI-GPT"]["GPT-Model"]
    OPENAI_MODEL_TEMPERATURE = int(config_json["OpenAI-GPT"]["Temperature"])
    OPENAI_MODEL_MAXTOKENS = min(2048, int(config_json["OpenAI-GPT"]["MaxTokens"]))
    
    CHATGLM_MODEL = config_json["ChatGLM"]["GPT-Model"]

    context_for_users = {}
    context_for_groups = {}

    GENERATE_PICTURE_ARG_PAT = re.compile("(\(|（)([0-9]+)[ \n\t]+([0-9]+)[ \n\t]+([0-9]+)(\)|）)")
    GENERATE_PICTURE_ARG_PAT2 = re.compile("(\(|（)([0-9]+)[ \n\t]+([0-9]+)[ \n\t]+([0-9]+)[ \n\t]+([0-9]+)(\)|）)")
    GENERATE_PICTURE_NEG_PROMPT_DELIMETER = re.compile("\n+")
    GENERATE_PICTURE_MAX_ITS = 200 #最大迭代次数
    
USE_OPENAIGPT = False
USE_CHATGLM = False
    
if config_json["OpenAI-GPT"]["Enable"]:
    print(f"Use OpenAI GPT Model({GlobalData.OPENAI_MODEL}).")
    USE_OPENAIGPT = True
elif config_json["ChatGLM"]["Enable"]:
    print(f"Use ChatGLM({GlobalData.CHATGLM_MODEL}) as GPT-Model.")
    chatglm_tokenizer = AutoTokenizer.from_pretrained(GlobalData.CHATGLM_MODEL, trust_remote_code=True)
    chatglm_model = AutoModel.from_pretrained(GlobalData.CHATGLM_MODEL, trust_remote_code=True)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        chatglm_model = chatglm_model.to('mps')
    elif torch.cuda.is_available():
        chatglm_model = chatglm_model.to('cuda')
    chatglm_model = chatglm_model.eval()
    USE_CHATGLM = True
    
app = flask.Flask(__name__)

# 这个用于放行生成的任何图片，替换掉默认的NSFW检查器，公共场合慎重使用
def run_safety_nochecker(image, device, dtype):
    print("警告：屏蔽了内容安全性检查，可能会产生有害内容")
    return image, None

sd_args = {
    "pretrained_model_name_or_path" : config_json["Diffusion"]["Diffusion-Model"],
    "torch_dtype" : (torch.float16 if config_json["Diffusion"].get("UseFP16", True) else torch.float32)
}

sd_pipe = StableDiffusionPipeline.from_pretrained(**sd_args)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
if config_json["Diffusion"]["NoNSFWChecker"]:
    setattr(sd_pipe, "run_safety_checker", run_safety_nochecker)

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    sd_pipe = sd_pipe.to("mps")
elif torch.cuda.is_available():
    sd_pipe = sd_pipe.to("cuda")
    
GPT_SUCCESS = 0
GPT_NORESULT = 1
GPT_ERROR = 2

def CallOpenAIGPT(prompts : typing.List[str]):
    try:
        res = openai.ChatCompletion.create(
            model=config_json["OpenAI-GPT"]["GPT-Model"],
            messages=prompts
        )
        if len(res["choices"]) > 0:
            return (GPT_SUCCESS, res["choices"][0]["message"]["content"].strip())
        else:
            return (GPT_NORESULT, "")
    except openai.InvalidRequestError as e:
        return (GPT_ERROR, e)
    except Exception as e:
        traceback.print_exception(e)
        return (GPT_ERROR, str(e))
    
def CallChatGLM(msg, history : typing.List[str]):
    try:
        resp, hist = chatglm_model.chat(chatglm_tokenizer, msg, history=history)
        if isinstance(resp, tuple):
            resp = resp[0]
        return (GPT_SUCCESS, resp)
    except Exception as e:
        return (GPT_ERROR, str(e))
    
def add_context(uid : str, is_user : bool, msg : str):
    if not uid in GlobalData.context_for_users:
        GlobalData.context_for_users[uid] = []
    if USE_OPENAIGPT:
        GlobalData.context_for_users[uid].append({
            "role" : "system",
            "content" : msg   
        }
        )
    elif USE_CHATGLM:
        GlobalData.context_for_users[uid].append(msg)
        
def get_context(uid : str): 
    if not uid in GlobalData.context_for_users:
        GlobalData.context_for_users[uid] = []
    return GlobalData.context_for_users[uid]
    

@app.route("/chat_clear", methods=['POST'])
def app_chat_clear():
    data = json.loads(flask.globals.request.get_data())
    GlobalData.context_for_users[data["user_id"]] = []
    print(f"Cleared context for {data['user_id']}")
    return ""

@app.route("/chat", methods=['POST'])
def app_chat():
    data = json.loads(flask.globals.request.get_data())
    #print(data)
    uid = data["user_id"]

    if not data["text"][-1] in ['?', '？', '.', '。', ',', '，', '!', '！']:
        data["text"] += "。"
        
    if USE_OPENAIGPT:
        add_context(uid, True, data["text"])
        #prompt = GlobalData.context_for_users[uid]
        prompt = get_context(uid)
        resp = CallOpenAIGPT(prompt=prompt)
        #GlobalData.context_for_users[data["user_id"]] = (prompt + resp)
        add_context(uid, False, resp[1])
        #print(f"Prompt = {prompt}\nResponse = {resp[1]}")
    elif USE_CHATGLM:
        #prompt = GlobalData.context_for_users[uid]
        prompt = get_context(uid)
        resp = CallChatGLM(msg=data["text"], history=prompt)
        add_context(uid, True, (data["text"], resp[1]))
    else:
        pass
    
    if resp[0] == GPT_SUCCESS:
        return json.dumps({"user_id" : data["user_id"], "text" : resp[1], "error" : False, "error_msg" : ""})
    else:
        return json.dumps({"user_id" : data["user_id"], "text" : "", "error" : True, "error_msg" : resp[1]})

@app.route("/draw", methods=['POST'])
def app_draw():
    data = json.loads(flask.globals.request.get_data())
    
    prompt = data["prompt"]

    i = 0
    for i in range(len(prompt)):
        if prompt[i] == ':' or prompt[i] == '：':
            break
    if i == len(prompt):
        return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, "error_msg" : "格式不对，正确的格式是：生成图片：Prompt 或者 生成图片(宽 高 迭代次数 [图片最大数量(缺省1)])：Prompt"})
    

    match_args = re.match(GlobalData.GENERATE_PICTURE_ARG_PAT2, prompt[:i])
    if not match_args is None:
        W = int(match_args.group(2))
        H = int(match_args.group(3))
        ITS = int(match_args.group(4))
        NUM_PIC = int(match_args.group(5))
    else:
        match_args = re.match(GlobalData.GENERATE_PICTURE_ARG_PAT, prompt[:i])
        if not match_args is None:
            W = int(match_args.group(2))
            H = int(match_args.group(3))
            ITS = int(match_args.group(4))
            NUM_PIC = 1
        else:
            if len(prompt[:i].strip()) != 0:
                return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, "error_msg" : "格式不对，正确的格式是：生成图片：Prompt 或者 生成图片(宽 高 迭代次数 [图片最大数量(缺省1)])：Prompt"})
            else:
                W = 768
                H = 768
                ITS = config_json.get('DefaultDiffutionIterations', 20)
                NUM_PIC = 1

    if W > 2500 or H > 2500:
        return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, "error_msg" : "你要求的图片太大了，我不干了～"})
    
    if ITS > GlobalData.GENERATE_PICTURE_MAX_ITS:
        return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, "error_msg" : f"迭代次数太多了，不要超过{GlobalData.GENERATE_PICTURE_MAX_ITS}次"})

    prompt = prompt[(i+1):].strip()

    prompts = re.split(GlobalData.GENERATE_PICTURE_NEG_PROMPT_DELIMETER, prompt)
    prompt = prompts[0]

    neg_prompt = None 
    if len(prompts) > 1:
        neg_prompt = prompts[1]

    print(f"Generating {NUM_PIC} picture(s) with prompt = {prompt} , negative prompt = {neg_prompt}")
    
    try:
        if NUM_PIC > 1 and torch.backends.mps.is_available():  #Apple silicon上的bug：https://github.com/huggingface/diffusers/issues/363
            return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, 
                "error_msg" : "单prompt生成多张图像在Apple silicon上无法实现，相关讨论参考https://github.com/huggingface/diffusers/issues/363"})

        images = sd_pipe(prompt=prompt, negative_prompt=neg_prompt, width=W, height=H, num_inference_steps=ITS, num_images_per_prompt=NUM_PIC).images[:NUM_PIC]
        if len(images) == 0:
            return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, "error_msg" : "没有产生任何图像"})
        filenames = []
        for i, img in enumerate(images):
            img.save(f"latest-{i}.png")
            filenames.append(f"latest-{i}.png")
        return json.dumps({"user_name" : data["user_name"], "filenames" : filenames, "error" : False, "error_msg" : ""})        

    except Exception as e: 
        return json.dumps({"user_name" : data["user_name"], "filenames" : [], "error" : True, "error_msg" : str(e)})

@app.route("/info", methods=['POST', 'GET'])
def app_info():
    return "\n".join([f"GPT模型：{config_json['OpenAI-GPT']['GPT-Model'] if USE_OPENAIGPT else config_json['ChatGLM']['GPT-Model']}", f"Diffusion模型：{config_json['Diffusion']['Diffusion-Model']}",
     "默认图片规格：768x768 RGB三通道", "Diffusion默认迭代轮数：20",
      f"使用半精度浮点数 : {'是' if config_json['Diffusion'].get('UseFP16', True) else '否'}",
      f"屏蔽NSFW检查：{'是' if config_json['Diffusion']['NoNSFWChecker'] else '否'}"])

if __name__ == "__main__":

    if USE_OPENAIGPT:
        openai.api_key = GlobalData.OPENAI_APIKEY

    app.run(host="0.0.0.0", port=11111)