from langchain_ollama import OllamaEmbeddings

def embedding(model="nomic-embed-text"):
    """
    Return an embedding object, chunks transformed into vectors

    :param model: model for embedding; default is nomic-embed-text
    :return: embedded object, chunks transformed into vectors
    """
    return OllamaEmbeddings(model=model)
