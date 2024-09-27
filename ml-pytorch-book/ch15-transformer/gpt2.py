from transformers import GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed

print("========================================")
print("Using GPT 2 to generate text")
generator = pipeline('text-generation', model='gpt2')
set_seed(123)

resp = generator("Hey readers, today is",
                 max_length=20,
                 num_return_sequences=5)

for i, r in enumerate(resp):
    print(f"Response {i + 1}: {r['generated_text']}")

print("========================================")
print("Using GPT 2 to Generate Features for training other models")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Let us encode this sentence"  # sentence with 5 words (sentence length)
encoded_input = tokenizer(text, return_tensors='pt')

model = GPT2Model.from_pretrained('gpt2')
output = model(**encoded_input)
# This is the last hidden layer weights of the model
# used as input features for other models
print("Batch size", output.last_hidden_state.shape[0])
print("Sentence length", output.last_hidden_state.shape[1])
print("Embedding length", output.last_hidden_state.shape[2])
# Now, we could apply this feature encoding to a given dataset and train a
# downstream classifier based on the GPT-2-based feature representation
