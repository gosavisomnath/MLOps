from huggingface_hub import HfApi, create_repo
import os

# Get Hugging Face Token from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face token.")

# Define the Hugging Face Space name
HF_SPACE_NAME = "your-hf-username/tourism-prediction-app" # <--- IMPORTANT: Replace with your Hugging Face username and desired space name

# Initialize HfApi
api = HfApi(token=HF_TOKEN)

print(f"Attempting to create or get Hugging Face Space: {HF_SPACE_NAME}")
try:
    # Create the repository if it doesn't exist
    # If it exists, this call will succeed without creating a new one
    repo_url = create_repo(repo_id=HF_SPACE_NAME, repo_type="space", private=False, exist_ok=True, token=HF_TOKEN)
    print(f"Hugging Face Space '{HF_SPACE_NAME}' ensured to exist or created: {repo_url}")
except Exception as e:
    print(f"Error creating or accessing Hugging Face Space: {e}")
    print("Please ensure your HF_TOKEN has write access and the space name is valid.")
    exit(1)

# Define the files to upload from the deployment folder
DEPLOYMENT_DIR = "tourism_project/deployment"
files_to_upload = [
    os.path.join(DEPLOYMENT_DIR, "app.py"),
    os.path.join(DEPLOYMENT_DIR, "requirements.txt"),
    os.path.join(DEPLOYMENT_DIR, "Dockerfile"),
]

print(f"Uploading files to Hugging Face Space '{HF_SPACE_NAME}'...")

for file_path in files_to_upload:
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}. Skipping upload.")
        continue
    
    try:
        # Upload each file
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path), # Upload with its original filename in the repo root
            repo_id=HF_SPACE_NAME,
            repo_type="space",
            token=HF_TOKEN,
        )
        print(f"Successfully uploaded {file_path} to {HF_SPACE_NAME}/{os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        print("Deployment failed for this file.")

print("\nDeployment process to Hugging Face Spaces complete.")
print(f"Your Streamlit app should now be deploying at: https://huggingface.co/spaces/{HF_SPACE_NAME}")
