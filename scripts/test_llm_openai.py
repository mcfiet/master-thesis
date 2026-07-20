import os
import sys
try:
    from openai import OpenAI
except ImportError:
    print("Error: The 'openai' library is not installed in this environment.")
    print("Please install it using: pip install openai")
    sys.exit(1)

# Base URL for the local LLM server
BASE_URL = "http://193.175.180.196:8000/v1"

def main():
    print("=== LLM Server Connection Tester (OpenAI SDK) ===")
    
    # Retrieve API key from environment variable
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("No LLM_API_KEY environment variable found.")
        api_key = input("Please enter your LLM API token/key: ").strip()
        if not api_key:
            print("Error: Token cannot be empty.")
            sys.exit(1)
            
    # Initialize the OpenAI client with the custom base URL and API key
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key
    )
    
    try:
        # 1. List available models
        print(f"\nFetching models from {BASE_URL}/models...")
        models_response = client.models.list()
        models = list(models_response.data)
        
        if not models:
            print("No models found on the server.")
            sys.exit(1)
            
        print("\nAvailable Models:")
        for idx, model in enumerate(models, 1):
            print(f"  {idx}. ID: {model.id} (Created by: {model.owned_by})")
            
        # Select the first model for testing
        selected_model = models[0].id
        print(f"\nSelected default model for test: {selected_model}")
        
        # 2. Test chat completion
        prompt = "Hallo! Kannst du mich hören? Bitte antworte kurz und bestätige, dass du funktionierst."
        print(f"\nSending test prompt to model '{selected_model}'...")
        print(f"Prompt: \"{prompt}\"")
        
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        answer = completion.choices[0].message.content
        print("\nResponse from LLM:")
        print("-" * 40)
        print(answer.strip())
        print("-" * 40)
        print("Success! The LLM server is responding correctly via OpenAI SDK.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you are connected to the Hochschulnetz (or VPN) and that your token is correct.")

if __name__ == "__main__":
    main()
