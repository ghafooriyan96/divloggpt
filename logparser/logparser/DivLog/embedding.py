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
parser.add_argument('-lines', type=int, default=20000, help='Number of lines to process (default is 20000)')
parser.add_argument('-batch_size', type=int, default=5, help='Number of logs to process per batch (default is 5)')
args = parser.parse_args()

# Set Azure OpenAI configurations
openai.api_key = args.key
openai.api_base = args.endpoint
openai.api_type = "azure"
openai.api_version = args.api_version

# Check if embeddings directory exists
if not os.path.exists("embeddings"):
    os.mkdir("embeddings")

# Function to get embeddings for a batch of logs
def get_log_embeddings_batch(batch_logs, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            # Send a batch of logs to the embedding model
            response = openai.Embedding.create(
                input=batch_logs,
                engine="text-embedding-ada-002"  # Use the embedding model
            )
            return [item['embedding'] for item in response['data']]  # Return the embeddings for the batch
        except openai.error.RateLimitError as e:
            wait_time = backoff_factor * (2 ** attempt)  # Exponential backoff
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None

# Load the raw log file (each line is a log entry)
input_file = args.input_file
print(f"Generating embeddings for {input_file} ...")

# Read the entire log file with UTF-8 encoding
with open(input_file, "r", encoding="utf-8") as file:
    logs = [line.strip() for line in file if line.strip()]

# Set the batch size for embedding requests
batch_size = args.batch_size

# Process the logs in batches and generate embeddings
embedding = {}
total_batches = len(logs) // batch_size + (1 if len(logs) % batch_size else 0)

for i in tqdm(range(0, len(logs), batch_size), desc="Processing Batches", total=total_batches):
    batch_logs = logs[i:i+batch_size]  # Get the current batch
    batch_labels = [log.split()[0] for log in batch_logs]  # Extract labels (first part of each log)
    batch_messages = [' '.join(log.split()[1:]) for log in batch_logs]  # Extract log messages (everything after the label)

    log_embeddings = get_log_embeddings_batch(batch_messages)
    if log_embeddings:
        for log_message, label, log_embedding in zip(batch_messages, batch_labels, log_embeddings):
            embedding[log_message] = {
                'label': label,
                'embedding': log_embedding
            }

# Extract the file name to use for saving the result
file_name = os.path.basename(input_file).split('.')[0]

# Save embeddings in JSON format
output_file = f"embeddings/{file_name}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embedding, f, separators=(',', ':'))

print(f"Embeddings saved to {output_file}")
