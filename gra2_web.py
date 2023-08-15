import gradio as gr
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch

max_length = 2048
top_p = 0.6
temperature = 0.95

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    model = AutoModel.from_pretrained("model", trust_remote_code=True, device_map='auto')
    model = PeftModel.from_pretrained(model, "output").half()
    model = model.eval()
    return tokenizer, model

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    if len(history) > 0:
        if len(history) > MAX_BOXES:
            history = history[-MAX_TURNS:]
        for i, (query, response) in enumerate(history):
            print(f"用户：{query}")
            print(f"AI：{response}")

    print(f"用户：{input}")
    print("AI正在回复：")
    responses = []
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p, temperature=temperature):
        query, response = history[-1]
        responses.append(response)
        print(f"AI：{response}")

    return history, responses

history = []

def chatbot(input_text):
    global history
    history, responses = predict(input_text, max_length, top_p, temperature, history)
    return responses[-1] if responses else ""

inputs = gr.inputs.Textbox(lines=10, label="用户命令输入")
outputs = gr.outputs.Textbox(label="AI回复")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="ChatGLM-6b 训练后演示").launch(share=True)
