from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# 选择 GPT-2 的预训练模型规模（如 "gpt2"、"gpt2-medium"、"gpt2-large" 或 "gpt2-xl"）
model_name = "gpt2-large"

# 下载预训练模型、分词器和配置文件
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

# 将模型、分词器和配置文件保存到本地
tokenizer.save_pretrained("Model")
config.save_pretrained("Model")
model.save_pretrained("Model")
