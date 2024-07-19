

# import whisper
# import os

# file_path = r"C:\Users\abc\Desktop\audio1.mp3"

# if __name__ == '__main__':
#     if not os.path.exists(file_path):
#         print(f"文件 {file_path} 不存在！")
#     else:
#         model = whisper.load_model("tiny")
#         try:
#             result = model.transcribe(file_path, fp16=False, language="English")
#             print(result["text"])
#         except Exception as e:
#             print(f"处理文件时发生错误：{e}")

import whisper
model = whisper.load_model("base")
file_path = r"C:\Users\abc\Desktop\audio1.mp3"
result1 = model.transcribe(file_path, fp16=False, language="English")
print(result1['text'])
model = whisper.load_model("tiny")
result2 = model.transcribe(file_path, fp16=False, language="English")
print(result2['text'])
model = whisper.load_model("medium")
result3 = model.transcribe(file_path, fp16=False, language="English")
print(result3['text'])


import openai

openaikey = "sk-ygDIx9uV40OaydpqetAb8dLMuzm6azyf8HeoVMyWdJdwdzNV"
#用的是学长的key
os.environ['OPENAI_API_KEY'] = openaikey
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.tech/v1'
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

from langchain_core.messages import HumanMessage


print(chat.invoke([HumanMessage(content="根据语法和合理性修改以下两段有错误歌词"+"1."+result1['text']+" 2."+result2['text'])]
))


