import requests
import json
import os

def load_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

first_article =  "articles/article1.txt"


#summarize 
def summarize_article(article_text, model_name="llama3.2"):
    url = "http://localhost:11434/api/chat"
    prompt = "Summarize this article in 3 bullet points."

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\n{article_text}"}
        ],
        "stream": True 
    }

    response = requests.post(url, json=payload, stream=True)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if "message" in data and "content" in data["message"]:
                full_response += data["message"]["content"]

    return full_response

#saving 
def save_summary(file_name, summary_text):
    os.makedirs("summaries", exist_ok=True)
    output_path = os.path.join("summaries", file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)



#gets all article
# Main logic
def process_all_articles():
    articles_dir = "articles"
    for filename in os.listdir(articles_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(articles_dir, filename)
            print(f"Processing {filename}...")

            article_text = load_article(file_path)
            summary = summarize_article(article_text)

            summary_filename = filename.replace(".txt", "_summary.txt")
            save_summary(summary_filename, summary)
            print(f"✔ Saved summary: summaries/{summary_filename}\n")

#viewing

if __name__ == "__main__":
    process_all_articles()


