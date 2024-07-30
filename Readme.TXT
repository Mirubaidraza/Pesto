Model Assessment README
Overview

The dataSet has been downloaded as the API wasn't working when accessed using the models.




This repository contains two different models for generating responses based on a dataset. Each model utilizes different approaches and architectures for handling the tasks. 
The two models included are:

Why GPT-3.5 Turbo or GPT-4 Wasn't Used
Due to API limitations and potential costs associated with using advanced models like GPT-3.5 Turbo or GPT-4, they were not used in this assessment. 
The GPT-2 model was chosen as it is more accessible and sufficient for the current task, though it does not always handle response generation perfectly.

GPT-2 Model
Description
The GPT-2 model was used for generating responses. The model was pre-trained on a dataset similar to the one provided, so no additional training was performed. 
The code was adjusted to fit the specific task requirements.

Issues and Solutions
Issue with Limiting Responses: Despite setting num_return_sequences=1, adjusting max_length, and setting do_sample=False, the model occasionally generates more than one responses.

Steps Taken to Address Issue:

Use num_return_sequences=1: Ensures only one response is generated.
Adjust max_length: Ensures responses are complete but not excessively long.
Set do_sample to False: Uses beam search for more deterministic results.
These adjustments did not completely resolve the issue.

Why GPT-3.5 Turbo or GPT-4 Wasn't Used
Due to API limitations and potential costs associated with using advanced models like GPT-3.5 Turbo or GPT-4, they were not used in this assessment. The GPT-2 model was chosen as it is more accessible
 and sufficient for the current task, though it does not always handle response generation perfectly.

Seq2Seq (T5) Model
Description
The Seq2Seq model, based on T5, is trained from scratch with the dataset provided. This model is used for generating responses based on input queries.

Data Preparation
For effective fine-tuning, the dataset needs to be in a specific format. Here's an example of the required JSON format:


Created a separate while in the desired format for the Seq2Seq model
json
Copy code
{
  "rows": [
    {"row": {"query": "how do I reset my password", "response": "Wie kann ich mein Passwort zurücksetzen?"}},
    {"row": {"query": "Can I get a refund for my purchase?", "response": "Kann ich eine Rückerstattung für meinen Kauf erhalten?"}}
    // Add more pairs here
  ]
}
Training
Tokenization: Both inputs and outputs are tokenized properly.
Dataset Preparation: Tokenized data is converted into a TensorFlow dataset.
Fine-Tuning: The model is adjusted with specific data.
Evaluation: Post fine-tuning, the model is evaluated to ensure proper functionality.
Key Points
Tokenization: Properly tokenize both inputs and outputs for each model.
Dataset Preparation: Convert tokenized data into a format suitable for training.
Fine-Tuning: Adjust each model with your specific data.
Evaluation: Ensure models are functioning correctly after training.