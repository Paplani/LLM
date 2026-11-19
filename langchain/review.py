from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv(find_dotenv())

gpt_model = init_chat_model(model="gpt-6-luna")


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0.0, le=1.0, description="감정의 강도")
    props: list[str] = Field(description="긍정적인 점의 목록")
    cons: list[str] = Field(description="부정적인 점의 목록")
    recommend: bool
    reply: str = Field(description="판매자 입장의 고객 답변 문장")


pydantic_parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
system = """
    당신은 상품을 판매하는 쇼핑몰의 운영자입니다. 
    사용자의 리뷰글을 보고 분석하는것이 당신의 역할입니다.
    항상 한국어로 답변해주세요.
    {format_instructions}   
    """

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        ("human", "{message}"),
    ]
).partial(format_instructions=pydantic_parser.get_format_instructions())

chain = prompt | gpt_model | pydantic_parser


def analyze(texts):

    texts = [line.strip() for line in texts.split("\n") if line.strip()]

    response = chain.batch([{"message": text} for text in texts])

    result = ""
    for i, (text, r) in enumerate(zip(texts, response), start=1):
        emoji = {"positive": "😄", "negative": "🤬", "neutral": "😑"}[r.sentiment]
        result += f"리뷰{i}\n"
        result += f"원 리뷰: {text}\n"
        result += f"감정: {r.sentiment} {emoji} :{r.score:.2f}\n"
        result += f"장점 : {', '.join(r.props) or '없음'}\n"
        result += f"단점 : {', '.join(r.cons) or '없음'}\n"
        result += f"추천 여부 : {'추천' if r.recommend else '비추천'}\n"
        result += f"판매자 답변 : {r.reply}\n\n"

    return result


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="리뷰 입력", lines=10, placeholder="배송이 빠르고 포장도 꼼꼼했어요."
    ),
    outputs=gr.Textbox(label="분석 결과", lines=20),
    title="상품 리뷰 분석",
)
demo.launch()
