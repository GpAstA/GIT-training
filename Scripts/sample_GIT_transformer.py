from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM

model_name = "microsoft/git-base-coco"

model = AutoModelForCausalLM.from_pretrained(model_name) # 画像からテキストを生成するタスクに（画像キャップション）に特化したGITモデル
processor = AutoProcessor.from_pretrained(model_name) #AutoProcesser：画像やテキストをモデルに適した形式に変換するために使われる。

# 画像のダウンロードと前処理
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = processor(images=image, return_tensors="pt").pixel_values # 

# テキストの前処理
prompt = "What is this?"
inputs = processor(
            prompt,
            image,
            return_tensors="pt",
            max_length=64
        )

sample = model.generate(**inputs, max_length=64)
print(processor.tokenizer.decode(sample[0]))
# two cats sleeping on a couch

print(model)