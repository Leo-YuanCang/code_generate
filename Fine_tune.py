from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json
from datetime import datetime

train_file = "answers_spe.json"  # 使用格式化问答数据集
#train_file = "output_spe.json"
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"runs/{train_file.replace('.','_')}/{current_time}"

# 加载预训练的 GPT-2 模型和分词器
config = GPT2Config.from_pretrained("Model")
tokenizer = GPT2Tokenizer.from_pretrained("Model")
tokenizer.sep_token = "</s>"
model = GPT2LMHeadModel.from_pretrained("Model", config=config)


class JsonDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.examples = []
        for item in data:
            # Extract the text from the JSON data
            question = item["instruction"]
            answer = item["output"]

            # Combine the question and answer with a separator token
            text = f"{question} {tokenizer.sep_token} {answer}"

            tokens = tokenizer.encode(text)
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# 预处理数据集
train_dataset = JsonDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    logging_dir=log_dir,  # 添加这一行，以将日志写入指定目录
)

# 微调模型
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# 保存微调后的模型和分词器到本地
model_save_path = train_file.replace('.','_')+"finetuned_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# 从本地加载微调后的模型和分词器
loaded_model = GPT2LMHeadModel.from_pretrained(model_save_path)
loaded_tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
