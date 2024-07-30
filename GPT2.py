import json
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load the data from the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract the responses from the nested structure
responses = [item['row']['response'] for item in data['rows']]

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Load the pre-trained model
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Prepare input text and attention mask
input_text = "How can I reset my password?"
input_ids = tokenizer.encode(input_text, return_tensors="tf")
attention_mask = tf.ones_like(input_ids)  # All ones, no padding in GPT-2

# Generate a longer response
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,  # Adjust max_length as needed
    num_beams=2,    # Beam search to find the most likely sequence
    early_stopping=True,  # Stop early when the model reaches the end of the sequence
    num_return_sequences=1,  # Ensure only one response is returned
    do_sample=False  # Use deterministic beam search instead of sampling
)

# Decode the response
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Post-process to remove repeated text
def trim_repeated_text(text):
    # Find the end of the first complete sentence
    sentences = text.split('. ')
    if len(sentences) < 2:
        return text
    
    first_sentence = sentences[0].strip() + '.'
    remaining_text = '. '.join(sentences[1:]).strip()

    # If remaining text starts with the first sentence, remove the repeated part
    if remaining_text.startswith(first_sentence):
        start_index = len(first_sentence) + 2  # +2 to account for the '. ' after the first sentence
        return text[:start_index] + remaining_text[start_index:].strip()
    
    return text

# Clean up the response text
clean_response = trim_repeated_text(response_text)

# Print the cleaned response
print(clean_response)
