from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Hello, I'm am conscious and"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

sample = model.generate(**input_ids, max_length=64)
print(tokenizer.decode(sample[0]))
# Hello, I'm am conscious and I'm a bit of a noob. I'm looking for a good place to start.

## modelの構造確認
print(model)

## 各層のパラメータ数を確認する
# 例えば、1層目の`self_attn`に関するパラメータを確認
first_layer_attn = model.model.decoder.layers[0].self_attn
print(first_layer_attn)