# 텍스트 => 요약, 키워드 추출, 분류, 감정분석, 어조 분석

from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv, find_dotenv


MODEL = "gpt-6-luna"
load_dotenv(find_dotenv())
client = OpenAI()


instructions = """
너는 텍스트 요약 도우미야. 너의 임무는 아래 주어진 텍스트 문장을 한국어로 요약하는거야.
- 요약 시 아래 사항을 반드시 반영해.
- 중복되 낸용은 생략하되, 반복되는 내용은 요약에서 더 강조해.
- 사례 중심보다는 개념과 주장 중심으로 요약해줘.
- 3줄 이내로 요약해줘.
- 불릿 기호 형식으로 작성해줘.
"""


def summarize_text(text):

    if text is None:
        return "텍스트를 입력해주세요."

    response = client.responses.create(
        model=MODEL, instructions=instructions, input=text
    )

    return response.output_text


demo = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=7, label="요약할 텍스트"),
    ],
    outputs=gr.Markdown(),
    title="OpenAI summarize program",
    description="OpenAI-Luna 기반 텍스트 요약 서비스",
)

demo.launch(debug=True)
