from dotenv import load_dotenv
import os
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from example import examples
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

load_dotenv()

os.makedirs("C:/big_data/12주차/farm/cache", exist_ok = True)

set_llm_cache(SQLiteCache(database_path = "C:/big_data/12주차/farm/cache/sqlite_cache1.db"))

def load_vector_store():

    farm_vs = Chroma(
        persist_directory = "./farm_db",
        embedding_function = OpenAIEmbeddings(model = "text-embedding-3-large"),
        collection_name = "f_db"
    )

    return farm_vs

def create_chain():

    llm = ChatOpenAI(model = "gpt-4.1", streaming = True)

    parser = StrOutputParser()

    tuple_examples = []

    for example in examples:
        tuple_examples.append(('human', example['question']))
        tuple_examples.append(('ai', example['answer']))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 농업 분야의 전문가입니다. 아래의 참고 문서를 바탕으로 사용자의 질문에 답변해 주세요."),
        *tuple_examples,
        MessagesPlaceholder(variable_name = 'history'),
        ("human", "참고 문서\n{context}\n\n질문\n{question}")
    ])

    chain = prompt | llm | parser

    return llm, chain

def get_answer(farm_vs, llm, query, chain, history):

    retriever = farm_vs.as_retriever()

    reorder = LongContextReorder()

    extractor = LLMChainExtractor.from_llm(llm)

    compressor = DocumentCompressorPipeline(
        transformers = [reorder, extractor]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = retriever
    )

    retriever_docs = compression_retriever.invoke(query)

    reference = "\n\n".join([doc.page_content for doc in retriever_docs])

    answer = chain.stream({"context" : reference, "question" : query, 'history' : history})

    return answer