import pandas as pd
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.etls.text_transforms import split_tables, text_cleanup_transforms

from ..settings import get_settings

settings = get_settings()


class CorpusItem(BaseModel):
    corpus_id: str
    text: str

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["CorpusItem"]:
        return [
            cls(
                corpus_id=row["corpus_id"],
                text=row["text"],
            )
            for _, row in df.iterrows()
        ]


def build_nodes(docs: list[CorpusItem]) -> list[TextNode]:
    nodes = []
    for doc in docs:
        text_without_tables, tables = split_tables(doc.text)
        transformed_text = text_cleanup_transforms(text_without_tables)
        doc_id = str(doc.corpus_id)
        chunks = transformed_text.split("\n\n")
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            nodes.append(
                TextNode(
                    text=chunk,
                    id_=f"{doc_id}:{i}",
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                    },
                )
            )
    return nodes


def vectorize_docs(
    docs: list[CorpusItem],
    collection_name: str,
) -> VectorStoreIndex:
    nodes = build_nodes(docs)
    qdrant_client = QdrantClient(url=settings.qdrant_url)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(
        api_base=settings.openai_emb_api_base,
        api_key=settings.openai_emb_api_key,
        model=settings.openai_emb_model,
        embed_batch_size=256,
        num_workers=3,
    )
    index = VectorStoreIndex.from_documents(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def vectorize_docs_from_file(
    file_path: str,
    collection_name: str,
) -> VectorStoreIndex:
    df = pd.read_csv(file_path)
    docs = CorpusItem.from_dataframe(df)
    return vectorize_docs(docs, collection_name)
