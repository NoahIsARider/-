

import whisper
import os

file_path = r"C:\Users\abc\Desktop\audio1.mp3"

if __name__ == '__main__':
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
    else:
        model = whisper.load_model("tiny")
        try:
            result = model.transcribe(file_path, fp16=False, language="English")
            print(result["text"])
        except Exception as e:
            print(f"处理文件时发生错误：{e}")

import whisper
model = whisper.load_model("base")
file_path = r"C:\Users\abc\Desktop\audio1.mp3"
result = model.transcribe(file_path, fp16=False, language="English")
print(result['text'])
model = whisper.load_model("tiny")
result = model.transcribe(file_path, fp16=False, language="English")
print(result['text'])
model = whisper.load_model("medium")
result = model.transcribe(file_path, fp16=False, language="English")
print(result['text'])

# import os
# import openai
#
# openaikey = "sk-ygDIx9uV40OaydpqetAb8dLMuzm6azyf8HeoVMyWdJdwdzNV"
# os.environ['OPENAI_API_KEY'] = openaikey
# os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.tech/v1'
# from langchain_openai import ChatOpenAI
#
# chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
#
# from langchain_core.messages import HumanMessage
#
# ans1 = "韩国环保运动联盟组织的示威活动强调全面禁止海豚表演,并敦促放归圈养海豚和白鲸,反映了公众对动物福利和生态保护的高度关注。动物表演长期以来因其对动物的身心健康影响而备受争议，环保组织的呼吁有助于推动国家和社会在尊重和保护动物权益方面取得进展。然而，如何科学合理地放归圈养动物是值得探讨的问题，需要专业机构和政府协作，共同找到最佳解决方案。"
# ans2 = "韩国环保运动联盟的示威活动体现了公众对动物福利和生态保护的日益重视。海豚表演长期以来因其对动物的身心健康影响而备受争议。环保组织的呼吁有助于推动政府和社会在尊重和保护动物权益方面取得进展。然而，放归圈养动物需要科学评估和周密计划，确保动物能够适应自然环境，同时也保护海洋生态平衡。"
# critic = "无害性 有效性"
# print(chat.invoke(
#     [
#         HumanMessage(
#             content="以下是两段不同的回答："+"1."+ans1+" 2."+ans2+" 请你按照以下标准分别给两个模型回答排序,排序结果按照 有效性：1>2 理由：，无害性：2>1 理由： 这种格式显示："+critic
#         )
#     ]
# ))


