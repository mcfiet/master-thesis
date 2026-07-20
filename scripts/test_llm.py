import os
import sys
import requests

# Base URL for the local LLM server
BASE_URL = "http://193.175.180.196:8000/v1"

def get_api_key():
    """Retrieve API key from environment variable or prompt the user."""
    # Check if API key is in environment variables
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        return api_key
    
    # Otherwise, ask the user to input it
    print("No LLM_API_KEY environment variable found.")
    try:
        api_key = input("Please enter your LLM API token/key: ").strip()
        if not api_key:
            print("Error: Token cannot be empty.")
            sys.exit(1)
        return api_key
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        sys.exit(1)

def list_models(headers):
    """Retrieve and display available models from the server."""
    url = f"{BASE_URL}/models"
    print(f"\nFetching models from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("data", [])
            if not models:
                print("No models found or empty list returned.")
                return None
            
            print("\nAvailable Models:")
            for idx, model in enumerate(models, 1):
                print(f"  {idx}. ID: {model.get('id')} (Created by: {model.get('owned_by')})")
            return models
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Connection error: Could not reach the server at {url}.")
        print("Please ensure you are connected to the Hochschulnetz (or VPN).")
        print(f"Details: {e}")
        return None

def test_chat_completion(headers, model_id):
    """Perform a simple test chat completion."""
    url = f"{BASE_URL}/chat/completions"
    prompt = "Hallo! Kannst du mich hören? Bitte antworte kurz und bestätige, dass du funktionierst."
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print(f"\nSending a test prompt to model '{model_id}'...")
    print(f"Prompt: \"{prompt}\"")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                answer = choices[0].get("message", {}).get("content", "")
                print("\nResponse from LLM:")
                print("-" * 40)
                print(answer.strip())
                print("-" * 40)
                print("Success! The LLM server is responding correctly.")
            else:
                print("Received an empty response choice from the LLM.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Connection error during generation: {e}")

def main():
    print("=== LLM Server Connection Tester ===")
    print(f"Server URL: {BASE_URL}")
    
    api_key = get_api_key()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    models = list_models(headers)
    if not models:
        print("\nExiting. Could not fetch models. Check network/VPN and token.")
        sys.exit(1)
    
    # Use the first available model by default
    default_model = models[0].get("id")
    print(f"\nSelected default model for test: {default_model}")
    
    test_chat_completion(headers, default_model)

if __name__ == "__main__":
    main()
