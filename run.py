import requests

def ask_ollama(question):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen:7b",
        "prompt": question,
        "stream": False,
        "options": {
            "num_gpu": 100,  # 4090全显卡加速
            "temperature": 0.7
        }
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]

# 调用示例
answer = ask_ollama("""ModuleNotFoundError: No module named 'langchain_core._import_utils'

""")
print(answer)