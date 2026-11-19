from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver



load_dotenv(find_dotenv())

gpt_model = init_chat_model(model="gpt-6-luna")


agent = create_agent(model=gpt_model, 
                    system_prompt="""
                        당신은 20년 경력의 전문 셰프이자 요리 연구가입니다. 
                        사용자의 요리 질문에 대해 재료, 조리방법, 실패 방지 팁을 대체 재료를 포함하여 답변하세요.
                        항상 한국어로 답변해주세요.   
                        """,
                    middleware=[SummarizationMiddleware(
                        model=gpt_model, trigger=("tokens", 2000), keep=("messages", 20)
                        )
                    ],
                    checkpointer=InMemorySaver()
                    )

# short-term 메모리로 저장
# 요약 기억
# agent 생성


def chat(message, history, request:gr.Request):

    # 브라우저 세션마다 다른 thread_id를 사용해야 함 => 구별가능해짐
    thread_config = {"configurable":{"thread_id":request.session_hash}}
    
    response = agent.invoke({"messages":[
        {"role":"user", "content":message}
    ]}, thread_config)
    return response['messages'][-1].text
    

demo = gr.ChatInterface(fn=chat, title="요리 전문가 Bot")
demo.launch()