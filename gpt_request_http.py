import requests as requests
from dotenv import load_dotenv
import os

# ohmygpt
# 在这里配置您在本站的API_KEY
load_dotenv('.env')

api_key = os.getenv('API_KEY')




"""
不用框架、只用http请求
"""
headers = {
    "Authorization": 'Bearer ' + api_key,
    "Content-Type": "application/json"
}


while True:
    question = input("question:")
    if question == "exit":
        break

    params = {
        "messages": [
            {
                "role": 'user',
                "content": question
            }
        ],
        # 如果需要切换模型，在这里修改
        "model": 'gpt-3.5-turbo'
    }

    response = requests.post(
        "https://cfwus02.opapi.win/v1/chat/completions",
        # "https://aigptx.top/v1/chat/completions",
        headers=headers,
        json=params,
        stream=False
    )

    # print(type(response))
    # print(response)

    res = response.json()
    # print(res)
    # print(res['choices'])

    res_content = res['choices'][0]['message']['content']
    print(res_content)
    print("====================")
