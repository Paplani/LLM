import gradio as gr
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model(model="gpt-6-luna", reasoning={"effort": "low"})


def process_pdf(pdf_file):

    if pdf_file is None:
        return ("PDF 파일을 업로드해주세요", "", "", "", "")

    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 첫 번째 페이지 내용
    first_page_content = docs[0].page_content

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 글자를 읽을 수 없는 PDF(이미지로 된 PDF 등)는 청크가 0개
    if not chunks:
        return (
            f"총 페이지 수: {len(docs)}",
            "",
            "총 청크 수 0",
            "PDF에서 글자를 읽을 수 없습니다 (이미지로 된 PDF일 수 있어요)",
            "",
        )

    # 총 청크 수
    total_chunks = len(chunks)

    # 첫 번째 청크 내용
    first_chunk = chunks[0].page_content

    # 첫 번째 청크의 메타데이터
    first_chunk_metadata = chunks[0].metadata

    return (
        f"총 페이지 수: {len(docs)}",
        first_page_content,
        f"총 청크 수 {total_chunks}",
        first_chunk,
        str(first_chunk_metadata),
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_chat(pdf_files, question):

    if pdf_files is None:
        return ("PDF 파일을 다시 업로드해주세요.", "")

    loader = PyPDFLoader(pdf_files)
    docs = loader.load()

    # 받은 내용 청킹하기
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # vector store에 저장 : 질문마다 새 빈 저장소 (다른 PDF와 섞이지 않게)
    # Chroma는 저장 폴더 없이도 같은 기본 컬렉션('langchain')에 계속 쌓여서 InMemory 사용
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=openai_embeddings,
    )

    # RAG(Retriever)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    retrieved_docs = retriever.invoke(question)

    formatted_retrieved_docs = format_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "다음 컨텍스트를 참고하여 질문에 답변해주세요.\n컨텍스트에 없는 내용은 모른다고 답하세요\n\n컨텍스트 : {context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = prompt | llm | StrOutputParser()

    answer = rag_chain.invoke(
        {"context": formatted_retrieved_docs, "question": question}
    )

    retrieved_text = ""
    for i, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        retrieved_text += f"[검색 문서 {i}] p.{page}\n\n"
        retrieved_text += f"근거 내용:\n{doc.page_content}\n\n"
        retrieved_text += f"{'=' * 50}\n\n"

    return (retrieved_text, answer)


with gr.Blocks() as demo:
    gr.Markdown("# PDF RAG 테스트")

    with gr.Tabs():
        with gr.Tab("1단계 - PDF & Chunk"):
            pdf_input_1 = gr.File(label="PDF 업로드", file_types=[".pdf"])
            run_btn_1 = gr.Button("분석 시작")
            page_output = gr.Textbox(label="총 페이지 수")
            first_page_output = gr.Textbox(label="첫번째 페이지 내용", lines=10)
            chunk_count_output = gr.Textbox(label="총 Chunk 수")
            first_chunk_output = gr.Textbox(label="첫번째 Chunk 내용")
            metadata_output = gr.Textbox(label="첫 번째 metadata 내용")

            run_btn_1.click(
                fn=process_pdf,
                inputs=pdf_input_1,
                outputs=[
                    page_output,
                    first_page_output,
                    chunk_count_output,
                    first_chunk_output,
                    metadata_output,
                ],
                api_name="greet",
            )

        with gr.Tab("2단계 - PDF Q&A"):
            pdf_input_2 = gr.File(label="PDF 업로드", file_types=[".pdf"])
            question_input = gr.Textbox(label="질문 입력")
            run_btn_2 = gr.Button("질문하기")

            retrieved_output = gr.Textbox(label="검색된 chunk", lines=10)
            answer_output = gr.Textbox(label="최종 답변", lines=10)

            run_btn_2.click(
                fn=rag_chat,
                inputs=[pdf_input_2, question_input],
                outputs=[retrieved_output, answer_output],
            )

demo.launch()
