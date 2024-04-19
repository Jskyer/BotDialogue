from transformers import pipeline

generator = pipeline("text-generation", model="./distilgpt2")
text = generator(
    "In this course, we will teach you how to",
    pad_token_id=generator.tokenizer.eos_token_id,
    max_length=30,
    num_return_sequences=2
)

print(text)
