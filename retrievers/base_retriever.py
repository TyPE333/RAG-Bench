
from abc import ABC, abstractmethod
from typing import List, Dict
from typing import Any

# Define the base class
class BaseRetriever(ABC):

    """
    Base class for all retrievers.
    This class defines the interface for all retrievers. It is not intended to be used directly.
    Instead, use one of the subclasses that implement the specific retrieval logic.
    This class provided the retrieval and indexing methods that are common to all retrivers.
    """

    @abstractmethod
    def index(self, corpus: List[Dict[str, str]]) -> None:
        """
        Build an internal index from the given corpus.
        The corpus is a list of documents, where each document is a dictionary with the following keys:
        - 'id': The unique identifier of the document.
        - 'text': The text of the document.
        """


        pass

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieve the top k documents from the index that are most relevant to the query.
        The query is a string, and k is the number of documents to retrieve.
        The method returns a list of dictionaries, where each dictionary contains the following keys:
        - 'id': The unique identifier of the document.
        - 'text': The text of the document.
        - 'score': The score assigned to the document by the retriever.
        """

        pass

