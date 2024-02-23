from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class Chain:
    """
    This class is used to create a rag chain or chain for the LLM model.
    """
    def __init__(self, llm):
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm

    def create_rag_chain(self, retriever):
        """
        This method is used to create a rag chain.
        retriever is the retriever that is used to retrieve the document.
        """
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
        )
        return chain

    def create_chain(self, output_parser):
        """
        This method is used to create a chain.
        """
        template = """{instruction}.{format_instruction}"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | output_parser
        return chain
