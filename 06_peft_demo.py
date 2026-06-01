# 1、导入相关的包
from peft import LoraConfig,get_peft_model
import os
os.environ["TENSORBOARD_LOGGING_DIR"]="logs/Qwen3-0.6B-LoRA"
# 2、构造一个LoraConfig对象
lora_config = LoraConfig(
    r=32, # 基于硬件资源和任务复杂度决定传递多少，这个值大，就会对资源要求更高，这个值小，对于复杂任务而言，可能会存在欠拟合的风险
    # 1、只对q_proj和v_proj两个参数矩阵调整
    # target_modules=["q_proj","v_proj"],
    # 2、扩展到其他的线性层
    # target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    # 3、更加简洁的写法：作用到所有的线性层上面
    target_modules="all-linear",
    lora_alpha=32, # 设置成2r,64
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

from transformers import AutoModelForCausalLM,AutoTokenizer
# 3、获取peft_model
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B")
peft_model = get_peft_model(model,lora_config)


from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# 4、准备数据：把原始数据，转换成SFTTrainer所需要的数据类型，
# 4.1 加载数据
datasets_dict = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})

# 4.2 使用map方法，对数据进行处理，处理成conversation格式
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

# 5、构造SFTConfig实例
config = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=12,
    num_train_epochs=1,
    #
    # max_steps=10
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    learning_rate=5e-5,
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
    output_dir="finetuned/Qwen3-0.6B-LoRA",

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
trainer.save_model("finetuned/Qwen3-0.6B-LoRA")