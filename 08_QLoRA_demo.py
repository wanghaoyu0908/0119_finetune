# 1、导包
from peft import prepare_model_for_kbit_training,LoraConfig,get_peft_model
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"]="logs/Qwen3-8B-QLoRA"

# 2、构建一个BitsAndBytesConfig对象
bitsandbytes_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# 3、加载模型时，传入config
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B",quantization_config=bitsandbytes_config)
# model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B")

model.to("cuda")

# 4、调用prepare_model_for_kbit_training对模型进行预处理
model = prepare_model_for_kbit_training(model)

# 5、接下来的操作，就和LoRA是一致的
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model,lora_config)


from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
# 6、加载数据
datasets_dict = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B")
from typing import Dict, List
def map(examples:Dict[str,List]):
    """
    对数据集进行处理
    输入：examples:原始数据，包含conversation_id，category等key
    输出：处理好的conversion数据，键是messages，值是一个列表，里面是两个json,分别表示human message 和 assistant message
    """
    conversations:List = examples["conversation"]
    new_conversations = []
    for conversation in conversations:
        # 第一层，conversation，是多个样本
        message_list = []
        for message in conversation:
            # 第二层，是一条数据当中的多个message
            for key, value in message.items():
                if key == "human":
                    message_list.append({"role":"user","content":value})
                else:
                    message_list.append({"role":"assistant","content":value})
        new_conversations.append(message_list)


            
    return {"messages":new_conversations}

mapped_datasets_dict = datasets_dict.map(function=map,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])

# 7、构造SFTConfig实例
config = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=12,
    num_train_epochs=1,
    #
    max_steps=1000,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # 评估和保存相关参数：
    eval_strategy="steps",
    eval_steps=100,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    output_dir="finetuned/Qwen3-8B-QLoRA",

    # 优化相关
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=500,
    
    # use_liger_kernel=True,
    # model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"}

    assistant_only_loss=True,
    chat_template_path="./chat_template.jinja"

)

trainer = SFTTrainer(
    model=peft_model,
    args=config,
    train_dataset=mapped_datasets_dict["train"],
    eval_dataset=mapped_datasets_dict["test"],
    processing_class=tokenizer
)


trainer.train()
trainer.save_model("finetuned/Qwen3-8B-QLoRA")