from email.mime import image
from cv2 import grabCut
import flask
import requests
import json
import openai
import re
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import argparse

ps = argparse.ArgumentParser()
ps.add_argument("--config", default="config.json", help="Configuration file")
args = ps.parse_args()

with open(args.config) as f:
    config_json = json.load(f)

class GlobalData:
   # OPENAI_ORGID = config_json[""]
    OPENAI_APIKEY = config_json["OpenAI-API-Key"]
    OPENAI_MODEL = config_json["GPT-Model"]
    OPENAI_MODEL_TEMPERATURE = 0.66
    OPENAI_MODEL_MAXTOKENS = 2048

    context_for_users = {}
    context_for_groups = {}

    GENERATE_PICTURE_ARG_PAT = re.compile("(\(|（)([0-9]+)[ \n\t]+([0-9]+)[ \n\t]+([0-9]+)(\)|）)")
    GENERATE_PICTURE_ARG_PAT2 = re.compile("(\(|（)([0-9]+)[ \n\t]+([0-9]+)[ \n\t]+([0-9]+)[ \n\t]+([0-9]+)(\)|）)")
    GENERATE_PICTURE_NEG_PROMPT_DELIMETER = re.compile("\n+")
    GENERATE_PICTURE_MAX_ITS = 200 #最大迭代次数

app = flask.Flask(__name__)
sd_pipe = StableDiffusionPipeline.from_pretrained(config_json["Diffusion-Model"], torch_dtype=torch.float32)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
if torch.backends.mps.is_available():
    sd_pipe = sd_pipe.to("mps")
elif torch.cuda.is_available():
    sd_pipe = sd_pipe.to("cuda")

def send_text_to_user(user_id : str, text : str):
    requests.post(url="http://localhost:11110/send_text", 
        data=json.dumps({"user_id" : user_id, "text" : text, "in_group" : False}))

def call_gpt(prompt : str):
    try:
        res = openai.Completion.create(
            model=GlobalData.OPENAI_MODEL,
            prompt=prompt,
            max_tokens=GlobalData.OPENAI_MODEL_MAXTOKENS,
            temperature=GlobalData.OPENAI_MODEL_TEMPERATURE)
        if len(res["choices"]) > 0:
            return res["choices"][0]["text"].strip()
        else:
            return ""
    except:
        return "上下文长度超出模型限制，请对我说\“重置上下文\"，然后再试一次"

@app.route("/chat_clear", methods=['POST'])
def app_chat_clear():
    data = json.loads(flask.globals.request.get_data())
    GlobalData.context_for_users[data["user_id"]] = ""
    print(f"Cleared context for {data['user_id']}")
    return ""

@app.route("/chat", methods=['POST'])
def app_chat():
    data = json.loads(flask.globals.request.get_data())
    #print(data)
    prompt = GlobalData.context_for_users.get(data["user_id"], "")

    if not data["text"][-1] in ['?', '？', '.', '。', ',', '，', '!', '！']:
        data["text"] += "。"

    prompt += "\n" + data["text"]

    if len(prompt) > 4000:
        prompt = prompt[:4000]

    resp = call_gpt(prompt=prompt)
    GlobalData.context_for_users[data["user_id"]] = (prompt + resp)

    print(f"Prompt = {prompt}\nResponse = {resp}")

    return json.dumps({"user_id" : data["user_id"], "text" : resp, "in_group" : False})

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
                ITS = 50
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


if __name__ == "__main__":
    #openai.organization = GlobalData.OPENAI_ORGID
    openai.api_key = GlobalData.OPENAI_APIKEY

    app.run(host="0.0.0.0", port=11111)