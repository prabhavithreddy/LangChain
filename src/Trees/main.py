import os

from langchain.output_parsers import PydanticOutputParser

from src.Trees.Chain import Chain
from src.Trees.Node import Node
from src.Trees.PlayerInformation import PlayerInformation
from src.Trees.PromptResponse import PromptResponse
from langchain.chat_models import AzureChatOpenAI
import warnings
warnings.filterwarnings(action="ignore")
os.environ["REQUESTS_CA_BUNDLE"] = r"../../ca-bundle-full.crt"

def build_tree(llm):
    """
    This function is used to build the tree. It takes the llm as input and returns the tree. We are trying to build a
    tree to get information about a cricket player. If not we can extend the tree to get other information as well
    """
    chain = Chain(llm)
    output_parser = PydanticOutputParser(pydantic_object=PromptResponse)
    tree = Node("Is the document is related to Cricket", chain, output_parser,
                left=Node("The given document is related to which sports", chain, output_parser,
                          left=Node("If the given document is not related to Football then say I am sorry", chain,
                                     output_parser),
                          right=Node("If the given document is related to Football then identify the player", chain, output_parser)
                          ),
                right=Node("Is the document about Rahul Dravid", chain, output_parser,
                           left=Node("Then which player is the document about", chain, output_parser),
                           right=Node("In Rahul Dravid's entire career how many runs he scored in Tests", chain,
                                      output_parser)))
    return tree


llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type=os.environ["OPENAI_API_TYPE"],
    temperature=0
)

url = "https://simple.wikipedia.org/wiki/Lionel_Messi"
tree = build_tree(llm)
player_information = PlayerInformation()

retriever = player_information.get_document_embeddings(url)
player_information.analyze(tree, retriever)
