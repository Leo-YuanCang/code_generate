import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr

# 加载经过微调的模型和分词器
model = GPT2LMHeadModel.from_pretrained("answers_spe_jsonfinetuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("answers_spe_jsonfinetuned_model")

def generate_answer(question):
    input_ids = tokenizer.encode(question, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # 创建全1的注意力掩码
    pad_token_id = tokenizer.eos_token_id  # 设置填充令牌ID为终止令牌ID
    output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=512, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).replace(question, '').strip()
    return answer.replace("</s>", "")


interface = gr.Interface(
    fn=generate_answer,
    inputs=gr.inputs.Textbox(lines=2, label="请输入问题："),
    outputs=gr.outputs.Textbox(label="答案："),
    title="GPT-2 问答模型",
    description="请输入一个问题，模型将生成一个答案。"
)

interface.launch(share=True)
