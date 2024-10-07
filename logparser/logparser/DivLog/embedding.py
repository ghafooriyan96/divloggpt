import openai
import json
import os
import pandas as pd
from tqdm import tqdm
import argparse
import time

# Argument parser for API key, endpoint, and version
parser = argparse.ArgumentParser(description="Runs embedding.py with Azure OpenAI's text-embedding-ada-002.")
parser.add_argument('-key', type=str, required=True, help='Azure OpenAI API key')
parser.add_argument('-endpoint', type=str, default="https://isu-agtech-fault-prediction.openai.azure.com/", help='Azure OpenAI API endpoint')
parser.add_argument('-api_version', type=str, default="2023-03-15-preview", help='Azure OpenAI API version')
parser.add_argument('-input_file', type=str, required=True, help='Path to the input CSV file with log data')
args = parser.parse_args()

# Set Azure OpenAI configurations
openai.api_key = args.key
openai.api_base = args.endpoint
openai.api_type = "azure"
openai.api_version = args.api_version

# Check if embeddings directory exists
if not os.path.exists("embeddings"):
   os.mkdir("embeddings")

'''
# Multiple Input files
input_dir = "../../data/loghub_2k/"
output_dir = "embeddings/"
log_list = ['BGL']
'''


# Function to get embeddings from the text-embedding-ada-002 model
def get_log_embedding(log, retries=5, backoff_factor=2):
   for attempt in range(retries):
       try:
           response = openai.Embedding.create(
               input=log,
               engine="text-embedding-ada-002"  # Use the embedding model
           )
           return response['data'][0]['embedding']  # Return the embedding (a vector of numbers)
       except openai.error.RateLimitError as e:
           wait_time = backoff_factor * (2 ** attempt)  # Exponential backoff
           print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
           time.sleep(wait_time)
       except Exception as e:
           print(f"Error processing log: {e}")
           return None
# Load the raw log file (each line is a log entry)
input_file = args.input_file
print(f"Generating embeddings for {input_file} ...")
'''
//////Read the log file and limit to the specified number of lines (default 2000)/////

with open(input_file, "r") as file:
   logs = [next(file).strip() for _ in range(args.lines) if next(file).strip()]
'''
# Read the entire log file with UTF-8 encoding
with open(input_file, "r", encoding="utf-8") as file:
    logs = [line.strip() for line in file if line.strip()]


# Process the logs and generate embeddings
embedding = {}
for log in tqdm(logs):
   log = log.strip()  # Remove any leading/trailing whitespace
   if log:  # Ensure it's not an empty line
       log_embedding = get_log_embedding(log)
       if log_embedding:
           embedding[log] = log_embedding  # Store the embedding (vector)

# Extract the file name to use for saving the result
file_name = os.path.basename(input_file).split('.')[0]

# Save embeddings in JSON format
output_file = f"embeddings/{file_name}.json"
with open(output_file, "w") as f:
   json.dump(embedding, f, separators=(',', ':'))

print(f"Embeddings saved to {output_file}")