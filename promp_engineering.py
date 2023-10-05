import os
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# Load the environment variables from the .env file
load_dotenv()

# Now you can access the environment variables as if they were loaded into your system's environment
api_key = os.getenv("API_KEY")

# Load the WN18RR test set from a the txt file available here https://github.com/villmow/datasets_knowledge_embedding/raw/master/WN18RR/text/test.txt
wn18rr_test_set = pd.read_csv("test.txt", sep="\t", header=None)

# Convert the test set to a list of triplets
wn18rr_test_set = [(triplet[0], triplet[1], triplet[2]) for triplet in wn18rr_test_set.values]

# Set up your OpenAI API credentials
openai.api_key = api_key

# Preprocess the WN18RR test set by hiding one entity in each triplet
# and create a list of triplets with hidden entities
test_triplets = [
    ("[HIDE_ENTITY]", relation, entity2) for (entity1, relation, entity2) in wn18rr_test_set
]

# Initialize a list to store the reciprocal ranks
reciprocal_ranks = []

# Loop through the test triplets and make API calls to ChatGPT
for triplet in tqdm(test_triplets):
    prompt = f"Given the relation '{triplet[1]}' and the entity '{triplet[2]}', find the missing entity: "
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,  # Adjust the max tokens as per your requirements
        n=1,  # Generate a single response
        stop=None,  # Let ChatGPT decide when to stop the completion
        temperature=0.7,  # Adjust the temperature as per your requirements
    )
    
    # Extract the generated entity from the API response
    generated_entity = response.choices[0].text.strip()
    
    # Compare the generated entity with the original hidden entity
    if generated_entity == triplet[0]:
        reciprocal_ranks.append(1)  # Correct prediction, rank = 1
    else:
        reciprocal_ranks.append(0)  # Incorrect prediction, rank = 0

# Calculate the Mean Reciprocal Rank (MRR)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

# Report the Mean Reciprocal Rank
print(f"Mean Reciprocal Rank: {mrr}")