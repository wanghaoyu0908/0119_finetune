"""
加载模型，进行推理
"""
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
# 1.通过命令行传参，加载指定路径的模型和tokenizer
from argparse import ArgumentParser
# 1.1、获取parser实例
parser = ArgumentParser()

# 1.2、添加参数
parser.add_argument("--model_path",type=str,default="Qwen/Qwen3")
parser.add_argument("--prompt",type=str)

# 1.3、解析参数
args = parser.parse_args()


model_path = args.model_path

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2、进行推理
prompt = args.prompt 
# 不能直接将prompt进行encode，传入到model.generate中，需要使用和训练时相同的Chat Template进行格式化
# input_ids= tokenizer.encode(prompt)
input_ids = tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt")["input_ids"].to("cuda")
# EOS TOKEN:用于控制自回归生成过程的终止，当模型在自回归生成过程中生成了EOS_Token，自回归生成就会结束。如果一直不生成，新生成的token总数达到了max_new_tokens之后，也会终止生成。
res = model.generate(input_ids = input_ids,max_new_tokens=500,eos_token_id=tokenizer.encode("<|im_end|>")[0])
# 3、对res进行处理
# 3.1、去除res中prompt的token_ids，仅保留新生成的token_ids
generated_token_ids = res[0][len(input_ids[0]):].tolist()

# 3.2、使用tokenizer，对generated_token_ids进行解码
generated_text = tokenizer.decode(generated_token_ids,skip_special_tokens = True)

print("新生成的文本：",generated_text)