# 1、导入Unsloth的包，Unsloth必须要放在最前面导入
from unsloth import FastLanguageModel
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from unsloth.chat_templates import train_on_responses_only
import os
os.environ["TENSORBOARD_LOGGING_DIR"]="logs/Qwen3-8B-QLoRA-Unsloth"
#2、加载模型
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,
    # 搭配这两个参数使用，让Unsloth加载本地模型
    use_exact_model_name=True,
    local_files_only = True
)

# 3、获取到peft_model
peft_model = FastLanguageModel.get_peft_model(
    model=model,
    r=32,
    target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_alpha=32,
    lora_dropout=0.05
)


datasets_dict = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})

from typing import Dict, List
def map(examples:Dict[str,List]):
    """
    对数据集进行处理，处理成message_list之后，调用tokenizer.apply_chat_template方法，获取到格式化字符串
    """
    conversations:List = examples["conversation"]
    new_conversations_text = []
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
                
        text = tokenizer.apply_chat_template(message_list,tokenize=False,add_generation_prompt = False)
        new_conversations_text.append(text)


            
    return {"text":new_conversations_text}

mapped_datasets_dict = datasets_dict.map(function=map,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])

# 5、构造SFTConfig实例
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
    output_dir="finetuned/Qwen3-8B-QLoRA-Unsloth",

    # 优化相关
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=500,
    
    # use_liger_kernel=True,
    # model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"}

    # assistant_only_loss=True, # 对于Unsloth而言，不需要传入这个参数
    chat_template_path="./chat_template.jinja"

)

trainer = SFTTrainer(
    model=peft_model,
    args=config,
    train_dataset=mapped_datasets_dict["train"],
    eval_dataset=mapped_datasets_dict["test"],
    processing_class=tokenizer
)


# 设置仅对assistant回答部分，计算损失
trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)


trainer.train()
peft_model.save_pretrained_merged("./finetuned/Qwen3-8B-SFT-unsloth-merged", tokenizer, save_method="merged_16bit")