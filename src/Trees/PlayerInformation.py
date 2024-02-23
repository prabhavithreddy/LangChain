from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import bs4


class PlayerInformation:
    """
    This class is used to get the information about the player. It makes use of a tree based prompt to get the
    information.
    """
    def format_docs(self, docs):
        """
        This method is used to format the documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def get_document_embeddings(self, url):
        """
        This method is used to get the document embeddings.
        It makes use of the AzureOpenAIEmbeddings to get the embeddings.
        """
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer("body")
            ),
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002"
        )
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        return retriever | self.format_docs

    def analyze(self, tree, retriever):
        """
        This method is used to analyze the tree.
        """
        root = tree
        while root:
            if root.invoke(retriever):
                root = root.right
            else:
                root = root.left
