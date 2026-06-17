from huggingface_hub import HfApi, create_repo
import os

# Get Hugging Face Token from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face token.")

# Define the Hugging Face dataset repository name
HUB_REPO_NAME = "tourism-package-prediction" # Ensure this matches the repo_id in data_preparation.py
HF_DATA_FILE = "tourism.csv"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv" # Path to the local dataset file

api = HfApi(token=HF_TOKEN)

print(f"Attempting to create or get Hugging Face dataset repo: {HUB_REPO_NAME}")
try:
    # Create the repository if it doesn't exist
    # If it exists, this call will succeed without creating a new one
    repo_url = create_repo(repo_id=HUB_REPO_NAME, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)
    print(f"Hugging Face dataset repo '{HUB_REPO_NAME}' ensured to exist or created: {repo_url}")
except Exception as e:
    print(f"Error creating or accessing Hugging Face dataset repo: {e}")
    print("Please ensure your HF_TOKEN has write access and the repo name is valid.")
    exit(1)

if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"Local dataset file not found at: {LOCAL_DATA_PATH}")

print(f"Uploading {HF_DATA_FILE} to Hugging Face Hub...")
try:
    api.upload_file(
        path_or_fileobj=LOCAL_DATA_PATH,
        path_in_repo=HF_DATA_FILE,
        repo_id=HUB_REPO_NAME,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"Successfully uploaded {HF_DATA_FILE} to {HUB_REPO_NAME}")
except Exception as e:
    print(f"Error uploading {HF_DATA_FILE}: {e}")
    print("Dataset upload failed.")
    exit(1)

print("Dataset registration to Hugging Face Hub complete.")
