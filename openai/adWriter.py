# 제품명, 브랜드명, 브랜드 핵심 가치, 제품 특징, 필수 포함 키워드 => 광고문구 1~2줄 5개 정도 추천

from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv, find_dotenv


MODEL = "gpt-6-luna"
load_dotenv(find_dotenv())
client = OpenAI()


instructions = """
너는 광고문구를 만들어주는 어시스턴트야. 너의 임무는 주어진 5개의 내용을 바탕으로 광고문구를 작성하는거야.
- 요약 시 아래 사항을 반드시 반영해.
- 제품명, 브랜드명, 브랜드 핵심 가치, 제품 특징, 필수 포함 키워드를 읽고 이것들을 반영해줘.
- 위의 키워드들을 바탕으로 광고문구를 작성해줘.
- 1~2줄 길이로 5개의 광고문구를 작성해줘.
"""


# 5개 전부 받아야함
def write_ad(product, brand, value, feature, keyword):

    if not all([product, brand, value, feature, keyword]):
        return "텍스트를 입력해주세요."

    content = f"""
제품명 : {product}
브랜드명 : {brand}
브랜드 핵심 가치 : {value}
제품 특징 : {feature}
필수 포함 키워드 {keyword}
"""

    response = client.responses.create(
        model=MODEL, instructions=instructions, input=content
    )

    return response.output_text


demo = gr.Interface(
    fn=write_ad,
    inputs=[
        gr.Text(label="제품명"),
        gr.Text(label="브랜드명"),
        gr.Text(label="브랜드핵심가치"),
        gr.Text(label="제품특징"),
        gr.Text(label="필수포함키워드"),
    ],
    outputs=gr.Markdown(),
    title="OpenAI 광고 문구 프로그램",
    description="OpenAI-Luna 기반 광고 문구 서비스",
)

demo.launch(debug=True)
