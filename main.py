from llm_client_scaleway import LLMClient
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化 LLMClient
client = LLMClient(
    api_url="https://api.scaleway.ai/e63882dd-2049-4317-a9ee-d03fc21c4ca8/v1/chat/completions",
    api_key=os.getenv("SCW_SECRET_KEY")
)

# 调用 API 获取响应
def ask_question(question):
    print(f"Asking: {question}")
    response = client.generate_response(user_message=question, stream=False)
    print("Response:", response)

if __name__ == "__main__":
    # 示例问题
    ask_question("What is the capital of China?")
