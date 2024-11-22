import requests
import json

class LLMClient:
    def __init__(self, api_url, api_key, model="llama-3.1-8b-instruct"):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"  # Bearer 格式
        }
        # print(f"Initialized with headers: {self.headers}")  # 调试用
        self.model = model

    def generate_response(self, user_message, system_message="You are a helpful assistant", 
                          max_tokens=512, temperature=0.7, top_p=0.7, 
                          presence_penalty=0, stream=True):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        # print(f"Payload: {payload}")  # 调试用
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), stream=stream)
            # print(f"Response status code: {response.status_code}")  # 调试用
            # print(f"Response text: {response.text}")  # 调试用
            response.raise_for_status()
            if stream:
                return self._stream_response(response)
            else:
                return self._parse_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def _stream_response(self, response):
        print("Generating response:\n")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line == "data: [DONE]":
                    break
                if decoded_line.startswith("data: "):
                    try:
                        data = json.loads(decoded_line[len("data: "):])
                        content = data["choices"][0]["delta"].get("content")
                        if content:
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
        print("\nResponse completed.")

    def _parse_response(self, response):
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
