from transformers import pipeline, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# 加载本地模型、分词器和配置文件
tokenizer = GPT2Tokenizer.from_pretrained("Model")
config = GPT2Config.from_pretrained("Model")
model = GPT2LMHeadModel.from_pretrained("Model", config=config)

# 创建文本生成管道
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # 将 device 设置为 0 以使用 GPU，如果使用 CPU，则设置为 -1

# 生成文本
input_text = "Once upon a time"
generated_text = text_generator(input_text, max_length=50, num_return_sequences=1, do_sample=True)[0]['generated_text']

print(generated_text)
