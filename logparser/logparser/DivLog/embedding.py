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
parser.add_argument('-endpoint', type=str, default="https://isu-agtech-fault-prediction.openai.azure.com/",
                    help='Azure OpenAI API endpoint')
parser.add_argument('-api_version', type=str, default="2023-03-15-preview", help='Azure OpenAI API version')
parser.add_argument('-input_file', type=str, required=True, help='Path to the input CSV file with log data')
parser.add_argument('-lines', type=int, default=20000, help='Number of lines to process (default is 20000)')
parser.add_argument('-batch_size', type=int, default=10, help='Number of logs to process per batch (default is 10)')
args = parser.parse_args()

# Set Azure OpenAI configurations
openai.api_key = args.key
openai.api_base = args.endpoint
openai.api_type = "azure"
openai.api_version = args.api_version

# Define the input file and print a message about the input
input_file = args.input_file
print(f"Generating embeddings for {input_file} ...")

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


# Function to process logs in batches and save embeddings to a local file or cloud
def process_logs_in_batches(file_path, batch_size=10, save_to_cloud=False):
    embedding = {}
    total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))  # Count total lines
    total_batches = total_lines // batch_size + (1 if total_lines % batch_size else 0)  # Calculate total batches

    with open(file_path, "r", encoding="utf-8") as file:
        with tqdm(total=total_batches, desc="Processing Batches") as progress_bar:  # Add progress bar
            while True:
                batch_logs = []
                for _ in range(batch_size):
                    line = file.readline()
                    if not line:
                        break
                    batch_logs.append(line.strip())
                if not batch_logs:
                    break  # End of file

                batch_labels = [log.split()[0] for log in batch_logs]  # Extract labels (first part of each log)
                batch_messages = [' '.join(log.split()[1:]) for log in batch_logs]  # Extract log messages

                log_embeddings = get_log_embeddings_batch(batch_messages)
                if log_embeddings:
                    for log_message, label, log_embedding in zip(batch_messages, batch_labels, log_embeddings):
                        embedding[log_message] = {
                            'label': label,
                            'embedding': log_embedding
                        }
                    # Save batch to local storage
                    save_embeddings(embedding, save_to_cloud)
                    embedding.clear()  # Clear memory after saving each batch

                progress_bar.update(1)  # Update progress bar after each batch


# Function to save embeddings locally or to cloud storage
def save_embeddings(embedding, save_to_cloud):
    file_name = os.path.basename(input_file).split('.')[0]  # Use input_file defined above
    output_file = f"embeddings/{file_name}.json"

    # Save to local disk
    with open(output_file, "a", encoding="utf-8") as f:  # Append to the file
        json.dump(embedding, f, separators=(',', ':'))

    # Save to cloud (you can implement cloud storage here, e.g., AWS S3 or Azure Blob Storage)
    if save_to_cloud:
        upload_to_cloud(output_file)


# Placeholder for cloud upload function (implement your cloud storage here)
def upload_to_cloud(output_file):
    # Implement the logic to upload the file to cloud (e.g., AWS S3, Google Cloud Storage, Azure Blob Storage)
    print(f"Uploading {output_file} to cloud...")
    # Example: You can use AWS S3, Google Cloud, or Azure SDK to upload
    pass


# Start processing logs in batches
process_logs_in_batches(input_file, batch_size=args.batch_size, save_to_cloud=False)

print("Embeddings processing completed.")
