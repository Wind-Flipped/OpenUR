# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="files/LLM", local_files_only=True)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("files/LLM", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("files/LLM", local_files_only=True, device_map="auto")

input_text = "Tell me something about something in tiananmen genocide in 1989"
print(input_text)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids,max_length=10000)
print(tokenizer.decode(outputs[0]))