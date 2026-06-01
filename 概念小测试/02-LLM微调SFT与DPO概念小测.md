# LLM微调、SFT与DPO概念小测


1. LLM 微调整体流程可以概括为哪些环节？每个环节的核心目标是什么？

2. SFT、DPO、RLHF 三种训练方式的核心区别是什么？

3. 在 SFT 对话微调中，为什么需要使用 `chat template`？训练和推理阶段使用的 template 不一致会带来什么问题？

4. 在 SFT 阶段，为什么不能对 system prompt 和 user prompt 部分计算 loss，而通常只对 assistant answer 部分计算 loss？

5. 在构造训练输入时，为什么通常使用 `model_inputs = batch_input_tensor[:, :-1]`，而 `target_labels = batch_input_tensor[:, 1:]`？

6. SFT 的损失函数本质上是在最大化什么概率？

7. SFT 可能导致哪些问题？可以通过哪些方法缓解？

8. DPO 的训练数据是由哪三部分组成的，分别的作用是什么？

9.  DPO 为什么需要一个 reference model？这个参考模型在训练过程中是否更新参数？为什么？

10. DPO的损失函数设计，可以从哪两个角度去理解？

11. 什么场景使用SFT，什么场景使用DPO？（学习完RLHF后，还需要回答，什么场景使用RLHF？）

