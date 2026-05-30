# LLM结构与Qwen3 模型概念小测

1. `Qwen3Model` 中 `tok_emb` 的作用是什么？输入 token id 经过它之后，张量形状会发生什么变化？

2. `TransformerBlock` 中为什么要在 Attention 和 FeedForward 前分别使用 `RMSNorm`？这属于 Pre-Norm 还是 Post-Norm 结构？

3. `GroupedQueryAttention` 和普通 Multi-Head Attention 相比，主要区别是什么？为什么 `K/V` 的 head 数可以少于 `Q` 的 head 数？

4. 在 `GroupedQueryAttention` 中，`keys.repeat_interleave(self.group_size, dim=1)` 的作用是什么？

5. Attention，为什么由MHA，发展出了MQA和GQA？
6. 在 `GroupedQueryAttention` 中，有哪些参数矩阵，分别的作用是什么
7. `compute_rope_params` 和 `apply_rope` 分别负责什么？RoPE 位置编码为什么是作用在 Query 和 Key 上，而不是 Value 上？


8. 什么是 `prefill` 阶段和 `decode` 阶段？`kv_cache` 是什么？在什么阶段会使用？

9. `emb_dim`、`n_heads` 和 `head_dim` 分别表示什么？

10. `Qwen3Model.forward()` 最后经过 `out_head` 得到的 `logits` 表示什么？

11. 在 `generate_text` 中，为什么每次生成新 token 后，只把 `next_input` 传回模型，而不是把完整的 `final_output` 再传一遍？

