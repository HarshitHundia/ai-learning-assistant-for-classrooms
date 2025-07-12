import requests
import json
import os
import asyncio # Still needed for running the async API call

# -------------------------------
# 1️⃣  ANSWER USING GEMINI API
# -------------------------------
async def get_gemini_answer(question_text):
    """Fetches an answer from the Gemini 2.0 Flash API."""
    print("🤔 Getting answer from Gemini (please wait)...")
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": question_text}]})

    payload = {"contents": chat_history}
    # IMPORTANT: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual Gemini API key.
    # When running this script locally, you must provide your own API key.
    # In the Canvas environment, this variable can be left as empty string ("")
    # as the key is automatically provided at runtime.
    apiKey = "AIzaSyAgrpayn0eNJy-hqm8OrOOGf6mFrzyaGPE" # <--- REPLACE THIS WITH YOUR ACTUAL API KEY

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            answer = result["candidates"][0]["content"].get("parts")[0].get("text", "No text found in response.")
            return answer
        else:
            print("❗ Warning: Gemini API response structure unexpected or content missing.")
            print("Full response:", result)
            return "Sorry, I couldn't get an answer from Gemini."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Gemini API: {e}")
        return "Sorry, I'm having trouble connecting to the answer service. Please check your API key and network connection."
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return "An unexpected error occurred while getting the answer."

# Main execution flow
if __name__ == "__main__":
    print("\n--- Text-Based Question Answer Module ---")
    print("Type 'exit' to quit.")
    print("-----------------------------------------\n")

    while True:
        # Get question from user input
        question = input("📝 Enter your question: ").strip()

        if question.lower() == 'exit':
            print("Exiting module. Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        # Get answer from Gemini API
        answer = asyncio.run(get_gemini_answer(question))

        # Print the answer
        print("🤖 Assistant Answer:", answer)
        print("\n-----------------------------------------\n")
