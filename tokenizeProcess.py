import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 指明检查点
checkpoint = "(bert_base)"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text = "hello world"
# 分词方法1
tensor1 = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# 分词方法2
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
# 通过模型发送多个句子，需要将ids放在一个列表里转为tensor
tensor = torch.tensor([ids])
print(tensor)
output = model(tensor)
print(output)