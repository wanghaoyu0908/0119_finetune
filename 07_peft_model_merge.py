# 1、导包
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from argparse import ArgumentParser
# 2、定义基座模型路径，LoRA适配器的路径
parser = ArgumentParser()
parser.add_argument("--base_model_path",type="str")
parser.add_argument("--adapter_path",type="str")
parser.add_argument("--save_path",type="str")
args = parser.parse_args()

base_model_path = args.base_model_path
adapter_path = args.adapter_path
save_path = args.save_path

# 3、加载基座模型和tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 4、加载LoRA适配器
peft_model = PeftModel.from_pretrained(model=base_model,model_id=adapter_path)

# 5、merge和保存
merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)