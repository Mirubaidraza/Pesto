import json
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Load and prepare the data
with open('dataseq.json', 'r') as file:
    data = json.load(file)

# Extract inputs and outputs
inputs = [item['row']['query'] for item in data['rows']]
outputs = [item['row']['response'] for item in data['rows']]

# Tokenize the data
train_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512, return_tensors='tf')
train_labels = tokenizer(outputs, truncation=True, padding=True, max_length=512, return_tensors='tf')

# Create a dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=train_encodings['input_ids'], attention_mask=train_encodings['attention_mask']),
    dict(labels=train_labels['input_ids'])
))

# Define a loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

def loss_fn(labels, logits):
    labels = tf.reshape(labels, (-1,))
    logits = tf.reshape(logits, (-1, logits.shape[-1]))
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(**inputs, labels=labels, return_dict=True)
        loss = loss_fn(labels, outputs.logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 3
batch_size = 4

# Prepare the dataset for batching
train_dataset = train_dataset.batch(batch_size)

for epoch in range(epochs):
    epoch_loss = 0
    for batch in train_dataset:
        inputs = batch[0]
        labels = batch[1]['labels']
        loss = train_step(inputs, labels)
        epoch_loss += loss.numpy()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataset)}")

# Define a function to generate a response given an input query
def generate_response(query):
    print(f"Generating response for query: {query}")
    inputs = tokenizer(query, return_tensors='tf', max_length=512, truncation=True)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_beams=4,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model with a specific input
test_query = "How do I reset my password?"
response = generate_response(test_query)
print(f"Response to '{test_query}': {response}")
