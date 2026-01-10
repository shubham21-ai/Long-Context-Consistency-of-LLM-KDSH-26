import time
from requests.exceptions import ConnectionError, ReadTimeout
from llama_index.retrievers.pathway import PathwayRetriever

# Initialize retriever
retriever = PathwayRetriever(host="127.0.0.1", port=8745, similarity_top_k=5)

# Define the query
query = 'consultants obligations'

# Retry loop
max_retries = 50  # Maximum number of retries
retry_delay = 2   # Initial delay in seconds

retrieved_results = None
attempts = 0

while attempts < max_retries:
    try:
        # Attempt to retrieve results
        retrieved_results = retriever.retrieve(str_or_query_bundle=query)
        break  # Exit loop if successful
    except (ConnectionError, ReadTimeout) as e:
        attempts += 1
        print(f"Attempt {attempts} failed: {e}")
        time.sleep(retry_delay)
        # retry_delay *= 2  # Exponential backoff

if retrieved_results is None:
    raise RuntimeError("Failed to connect after multiple retries.")

# Process retrieved results
for result in retrieved_results:
    print(result)
    print('------------')
