import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None

# Supported embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM produces 384-dimensional vectors
EMBEDDING_DIMENSIONS = 384


# Get embedding dimension
def get_embedding_dimension(model_name: str | None = None) -> int:

    name = (
        model_name or settings.embedding_model or EMBEDDING_MODEL
    ).strip().lower()

    if "all-minilm-l6-v2" in name:
        return EMBEDDING_DIMENSIONS

    raise ValueError(
        f"Unsupported embedding model: {name}. "
        f"Use {EMBEDDING_MODEL}"
    )


# Create HuggingFace embeddings
def get_embeddings():

    global _embeddings

    if _embeddings is None:

        model_name = settings.embedding_model

        # Validate model
        get_embedding_dimension(model_name)

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    return _embeddings


# Create or verify Pinecone index
def ensure_index():
    check_embedding_model()

    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing"
        )

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    index_names = pc.list_indexes().names()

    # Check existing index
    if settings.pinecone_index_name in index_names:

        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = index_info.dimension

        # Check index dimension
        if current_dimension != desired_dimension:

            raise RuntimeError(
                f"Pinecone index dimension mismatch. "
                f"Existing: {current_dimension}, "
                f"Required: {desired_dimension}. "
                f"Create a new Pinecone index with "
                f"dimension {desired_dimension}."
            )

    else:

        # Create new index
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until index is ready
        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:

            time.sleep(1)

    return pc.Index(
        settings.pinecone_index_name
    )


# Get Pinecone vector store
def get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


# Get retriever
def get_retriever():

    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# Add documents to Pinecone
def add_documents(chunks):

    store = get_vectorstore()

    return store.add_documents(chunks)

print("Current embedding model:", settings.embedding_model)



# Check embedding model configuration
def check_embedding_model():

    model_name = settings.embedding_model

    print("\n========== MODEL CHECK ==========")
    print("Embedding Model:", model_name)

    try:
        dimension = get_embedding_dimension(model_name)

        print("Embedding Dimension:", dimension)
        print("Model Configuration: Correct")

    except ValueError as error:
        print("Model Configuration: Incorrect")
        print("Error:", error)

    print("================================\n")