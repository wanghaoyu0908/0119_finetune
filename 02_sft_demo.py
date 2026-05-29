from transformers import AutoTokenizer
from dataclasses import dataclass
# 1、传入模型目录，加载tokenizer
model_path = "model/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

@dataclass
class SFTConfig:
    batch_size:int=4
    min_learning_rate:float=5e-6
    max_learning_rate:float = 5e-5
    warmup_step:int = 500

    log_dir:str = "logs/Qwen3-0.6B-SFT"
    log_iter:int = 100

    save_dir:str = "finetuned/Qwen3-0.6B-SFT"
    train_data_size:int = 20000

# 使用chat template，对数据集当中的数据进行处理
def get_train_data(config:SFTConfig):

    from datasets import load_dataset
    dataset = load_dataset("data/ultrachat_200k")
    train_sft_data:list = dataset["train_sft"]
    train_data = []
    for i in range(config.train_data_size):
        message_list = train_sft_data[i]["messages"]
        message_list.insert(0,{"role": "system", "content": "你是一个智能助手"})
        # tokenized_data: dict，包含了input_ids和attention_mask
        tokenized_data = tokenizer.apply_chat_template(message_list,tokenize=True,truncation=True,max_length = 2500)
        train_data.append(tokenized_data)
    
    return train_data

from transformers import PreTrainedTokenizerFast
import torch
from typing import List
def create_answer_mask(input_ids,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    """
    
    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵
    answer_mask = torch.zeros_like(input_ids)

    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    for idx,ids in enumerate(input_ids):
        # 获取到所有的eos_position
        # 假设有一条样本，总共有15个token，第一个<|im_end|>索引位置是5， 第二个是10，第三个15
        # 此时返回的eos_position为[5,10,15]
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()

        # 排除第一个eos_position: 第一个对应的是system prompt
        # 得到的eos_position为[10,15]
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[11,21]
        assistant_ends:[16,26]
    """

    use_ends = [pos+1 for pos in eos_positions[::2]]
    assistant_ends = [pos+1 for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[11,21]，assistant_ends:[16,26]

    user_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    assistant_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    要想获取到assistant的回答的起始位置，就需要再跳过\n,<|im_start|>,assistant 这三个token，所以需要加3.
    要想获取到assistant的回答的结束位置，就需要往前跳一个<|im_end|>，所以需要减1.
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    # if num_user_turns == num_assistant_turns:
    for user_end,assistant_end in zip(user_ends,assistant_ends):
        answer_start = user_end + 3
        answer_end = assistant_end - 1
        mask[answer_start:answer_end] = 1

# 根据SFT损失计算公式来写具体损失计算的方法/函数
def compute_loss(output_logits,labels,assistant_answer_mask):
    """
    计算当前批次的平均损失
    output_logits: 模型前向传播输出的logits, shape为：(batch_size, seq_len, vocab_size)
    labels: 真实标签，shape为：(batch_size, seq_len)
    assistant_answer_mask: 助手回答的mask码，shape为：(batch_size, seq_len)
    """
    # 1、计算logits的softmax，得到对数概率分布
    # log_probs: shape为:(batch_size, seq_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(output_logits,dim=-1)

    # 2、从log_probs概率分布中找到，模型输出真实标签对应的对数概率是多少
    answer_log_probs = torch.gather(log_probs,dim=-1,index=labels.unsqueeze(-1))
    # answer_log_probs: shape为:(batch_size, seq_len, 1)
    answer_log_probs = answer_log_probs.squeeze(-1)
    # 3、得到负对数概率
    answer_log_probs = answer_log_probs * (-1)

    # 4、使用assistant_answer_mask对answer_log_probs进行mask操作
    masked_answer_log_probs = answer_log_probs * assistant_answer_mask

    # 5、计算平均损失
    average_loss = masked_answer_log_probs.sum()  / assistant_answer_mask.sum()

    return average_loss

import numpy as np
def cosine_decay(current_step,total_step, min_lr, max_lr,warmup_step):
    """
    实现一个带warmup的cosine decay学习率调度器
    """
    if current_step < warmup_step:
        return current_step * max_lr / warmup_step
    else:
        progress = (current_step - warmup_step) / (total_step - warmup_step)
        return min_lr + (max_lr -min_lr)*(1+np.cos(np.pi * progress)) * 0.5
    
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import tqdm
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base/")



def train(config:SFTConfig):
    """
    手写主训练循环代码
    """
    # 1、获取模型，将模型置为训练模式，将模型放到cuda上
    model.train()
    model.to("cuda")
    # 2、获取到训练数据
    train_data = get_train_data(config)
    # 假设train_data最后剩余的数据不足batch_size，剩余范围是1到batch_size-1 ，再加上batch_size-1，batch_size 到 2batch_size-2
    # 假设，train_data是 10条数据，batch_size是4，10+3=13 13//4 = 3
    total_steps = (len(train_data)+ config.batch_size -1) // config.batch_size

    # 3、构造优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.min_learning_rate)

    # 4、日志记录
    writer = SummaryWriter(log_dir = config.log_dir)
    loss_list= []
    progress_bar = tqdm.tqdm(total=total_steps,desc="step")
    print("开始训练")
    for step in range(total_steps):
        
        # 1、构造一个batch的数据
        # 1.1 获取一个batch的数据
        batch_data = train_data[step* config.batch_size:(step+1) * config.batch_size]

        # 1.2 对数据进行padding

        batch_max_len = max([len(seq["input_ids"]) for seq in batch_data])
        padded_seq= []
        for seq in batch_data:
            current_seq_len = len(seq["input_ids"])
            padding_length = batch_max_len - current_seq_len

            seq["input_ids"].extend([tokenizer.pad_token_id] * padding_length)
            padded_seq.append(seq["input_ids"])
        
        # 1.3 从padded_seq中构造模型前向传播输入的input_ids和labels
        padded_seq_tensors = torch.tensor(padded_seq,dtype=torch.long).to("cuda")

        input_ids = padded_seq_tensors[:,:-1]
        labels = padded_seq_tensors[:,1:]

        # 1.4 基于input_ids构建一个assistant_answer_mask
        assistant_answer_mask = create_answer_mask(input_ids,tokenizer)

        if assistant_answer_mask.sum().item() == 0:
            print("当前batch中，没有assistant回答的部分")
            progress_bar.update(1)
            continue

        # 2、模型前向传播
        output_logits = model(input_ids).logits

        # 3、计算损失，反向传播，计算梯度
        loss = compute_loss(output_logits, labels, assistant_answer_mask)

        loss.backward()
        loss_list.append(loss.item())
        # 4、更新模型参数
        # 4.1 通过学习率调度器，获取到当前step的学习率
        current_step_learning_rate = cosine_decay(step,total_steps,min_lr=config.min_learning_rate,max_lr=config.max_learning_rate,warmup_step=config.warmup_step)
        writer.add_scalar("learning_rate",current_step_learning_rate,step)
        # 4.2 更新优化器的学习率
        optimizer.param_groups[0]["lr"] = current_step_learning_rate
        # 4.3 更新模型参数
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{loss_list[-1]:.4f}", lr=f"{current_step_learning_rate:.2e}")
        # 5、记录日志
        should_log = step % config.log_iter == 0
        if should_log:

            average_loss = np.mean(loss_list[-config.log_iter:])
            writer.add_scalar("train_loss",average_loss,step)

    model.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)
    print("模型训练完成")



if __name__ == "__main__":
    config = SFTConfig()
    train(config)