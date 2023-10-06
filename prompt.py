import os
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time
import ast

# Load the environment variables from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# Set up OpenAI API credentials
openai.api_key = api_key

# Load the WN18RR test set from a the txt file available here https://github.com/villmow/datasets_knowledge_embedding/raw/master/WN18RR/text/test.txt
wn18rr_test_set = pd.read_csv("test.txt", sep="\t", header=None)

# Convert the test set to a list of triplets
test_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in wn18rr_test_set.values]

# Chunk the test triplets into batches of size 50
batch_size = 50
num_batches = (len(test_triplets) + batch_size - 1) // batch_size
triplet_batches = [
    test_triplets[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)
]

# Initialize lists to store responses
responses2 = []
generated_entities2 = []

# Loop through the test triplets and make API calls to ChatGPT
for triplet_batch in tqdm(triplet_batches):
    # Open files in append mode
    responses2_file = open('results/responses2.txt', 'a')
    generated_entities2_file = open('results/generated_entities2.txt', 'a')

    # prompt = "Consider the relation prediction task, in the following lines the given entity comes first and the relation comes second after a space. Missing entities should have same format as given entities. Give me all the missing entities in a single array with this format:\nmissing entities:  ['me1', 'me2', ...]"
    prompt = "I have relation prediction task, consider these entities:"
    for triplet in triplet_batch:
        prompt += f'\n{triplet[0]}'
    prompt += "\nAnd also these relation respectively"
    for triplet in triplet_batch:
        prompt += f'\n{triplet[1]}'
    prompt += "Find just missing entities (NOT THE RELATION OR OTHER FALSY WORDS!) respectively and give them to me in a single array with this format:\nmissing entities:  ['', '', ...]"

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3100,  # Adjust the max tokens as per your requirements
        n=1,  # Generate a single response
        stop=None,  # Let ChatGPT decide when to stop the completion
        temperature=0.6,  # Adjust the temperature as per your requirements
    )
    
    # Extract the generated entity from the API response
    generated_entity = response.choices[0].text.strip()
    
    responses2.append(response)
    generated_entities2.append(generated_entity)

    responses2_file.write(str(response)+ '\n')
    generated_entities2_file.write(str(generated_entity)+ '\n')

    # Close files
    responses2_file.close()
    generated_entities2_file.close()

    # Time sleep for handling ChatGPT request rate limit
    time.sleep(19)

results = []
for i, ge in enumerate(generated_entities2):
    # Extract the array from the text
    start_index = ge.find("[")
    end_index = ge.find("]")
    array_str = ge[start_index:end_index+1]
    if end_index == -1:
        array_str = '[]'
    # Convert the string representation of the array to a Python list
    array = ast.literal_eval(array_str)

    # Append the extracted array
    results.append(array)