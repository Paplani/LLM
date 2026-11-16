# 텍스트 => 요약, 키워드 추출, 분류, 감정분석, 어조 분석

from ollama import Client
import gradio as gr


OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3.5:9b"

client = Client(host=OLLAMA_HOST)


task_dict = {
    "요약": "다음 글을 핵심 위주로 요약해줘.",
    "분류": "다음 글을 주제와 문서 유형을 분류해 줘.",
    "키워드 추출": "다음 글에서 키워드를 추출해줘.",
    "감정 분석": "다음 글에서 느껴지는 감정을 분석해줘.",
    "어조 분석": "다음 글에서 느껴지는 어조를 분석해줘.",
}

SYSTEM = "너는 텍스트 분석 전문가야. 한국어로 답변해줘"


def analyze(text, task):

    # 요약
    # {"role":"user","content":f"다음 글을 핵심 위주로 요약해줘. {text}"}
    prompt = task_dict[task]
    content = f"{prompt}\n\n{text}"

    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        stream=False,
        think=False,
    )

    return response.message.content


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(lines=7, label="분석할 텍스트"),
        gr.Dropdown(
            choices=["요약", "키워드 추출", "분류", "감정 분석", "어조 분석"],
            value="요약",
            label="분석작업",
        ),
    ],
    outputs=gr.Markdown(),
    title="AI Text Analyzer",
    description="Qwen3.5:9B 기반 텍스트 분석 서비스",
)

demo.launch(debug=True)
