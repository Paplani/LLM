# 글쓰기 도우미 서비스
# 격식체, 친근한 말투, 비지니스 문체, 간결한 문체, 전문적인 문체

from ollama import Client
import gradio as gr


OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3.5:9b"

client = Client(host=OLLAMA_HOST)


literary_dict = {
    "격식체": "다음 글을 공식적인 자리에 어울리는 격식체로 바꿔줘.",
    "친근한 말투": "다음 글을 친구에게 말하듯 친근하고 부드러운 말투로 바꿔줘.",
    "비즈니스 문체": "다음 글을 업무 메일에 쓸 수 있는 정중한 비즈니스 문체로 바꿔줘.",
    "간결한 문체": "다음 글을 군더더기 없이 짧고 간결한 문체로 바꿔줘.",
    "전문적인 문체": "다음 글을 보고서에 어울리는 전문적인 문체로 바꿔줘.",
}

SYSTEM = (
    "너는 글쓰기 도우미야. 한국어로 답변해줘. "
    "원문의 의미는 유지하고, 설명 없이 변환된 문장만 출력해줘."
)


def analyze(text, task):

    # 요약
    # {"role":"user","content":f"다음 글을 핵심 위주로 요약해줘. {text}"}
    prompt = literary_dict[task]
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
        gr.Textbox(lines=7, label="원본 문장"),
        gr.Radio(
            choices=list(literary_dict.keys()),
            value="격식체",
            label="분석작업",
        ),
    ],
    outputs=gr.Textbox(lines=7, label="변환 결과 문장"),
    title="AI Writing Assistant",
    description="Qwen3.5:9B 기반 글쓰기 도우미 서비스",
)

demo.launch(debug=True)
