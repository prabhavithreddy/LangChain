from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.Trees.Chain import Chain


class Node(object):
    """
    This class is used to create a node for the tree.
    It contains the instruction, chain, output_parser, left and right.
    """
    def __init__(self, instruction: str, chain: Chain, output_parser: PydanticOutputParser, left=None, right=None):
        self.instruction = instruction
        self.chain = chain
        self.left = left
        self.right = right
        self.output_parser = output_parser

    def invoke(self, retriever) -> bool:
        """
        This method is used to invoke the chain using a prompt.
        We can then use the response to decide which node to go to.
        """
        rag_output = self.chain.create_rag_chain(retriever).invoke(self.instruction)
        template = """{instruction}.{format_instruction}"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = self.chain.create_chain(self.output_parser)
        response = chain.invoke(
            {"instruction": "Based on the previous question return the answer" + "\n" + rag_output.content,
             "format_instruction": self.output_parser.get_format_instructions()})
        print(response)
        return response.Result
