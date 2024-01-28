from pydantic import validator, ValidationError, BaseModel
from typing import Set, List, Union, Any, Callable, Dict, Generator, List, Optional, Type, Sequence
import toml
import requests
from urllib.parse import urlparse
import os
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path

from llama_index.schema import Document, BaseNode, NodeWithScore, TextNode
from llama_index import ServiceContext, SimpleDirectoryReader
from pathlib import Path
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index import SimpleDirectoryReader
from llama_index.llms import LLM
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor
)
from llama_index.ingestion import IngestionPipeline
from llama_index import get_response_synthesizer
from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import PDFReader
from llama_index.vector_stores import ChromaVectorStore, WeaviateVectorStore
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.types import (
    VectorStoreInfo,
    MetadataInfo,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)

import weaviate

from phages.modules.extractors import get_citation
from phages.modules.answer import Answer
from phages.modules.postprocessor import SummaryAndScoreNodePostProcessor
from phages.modules.prompts import ask_llm_prompt, answer_prompt

class FullPDFReader(PDFReader):
    """Full PDF parser with default full document return."""

    def __init__(self) -> None:
        """
        Initialize FullPDFReader with return_full_document set to True by default.
        """
        super().__init__(return_full_document=True)

# for SimpleDirectoryReader file_extractor param
CUSTOM_FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": FullPDFReader(),
}


class SourceType(BaseModel):
    source: Union[Path, str]

    @validator('source', pre=True, always=True)
    def validate_source(cls, value):
        if isinstance(value, Path) or (isinstance(value, str) and os.path.exists(value)):
            return str(value)
        elif isinstance(value, str) and urlparse(value).scheme in ('http', 'https'):
            return value
        raise ValueError("The source must be a valid URL or an existing local file path.")

class Library():
    documents: List[Document] = []
    nodes: List[BaseNode] = []
    docs_index: VectorStoreIndex | None
    nodes_index: VectorStoreIndex | None
    service_context: ServiceContext = ServiceContext.from_defaults()
    client: weaviate.Client | None = None

    # class Config:
    #     arbitrary_types_allowed = True

    def __init__(self):
        self._initialize_storage()

    def withLLM(self, llm: LLM):
        self.service_context = ServiceContext.from_service_context(self.service_context, llm=llm)
        return self

    def __del__(self):
        if self.client is not None:
            self.client.close()

    def _initialize_storage(self):
        config = toml.load("../config.toml")
        WEAVIATE_PERSISTENCE_DATA_PATH = config["paths"]["WEAVIATE_PERSISTENCE_DATA_PATH"]
        WEAVIATE_BINARY_PATH = config["paths"]["WEAVIATE_BINARY_PATH"]
        os.makedirs(WEAVIATE_PERSISTENCE_DATA_PATH, exist_ok=True)
        os.makedirs(WEAVIATE_BINARY_PATH, exist_ok=True)

        # connect to your weaviate instance
        self.client = weaviate.Client(
            embedded_options=weaviate.embedded.EmbeddedOptions(
                persistence_data_path=WEAVIATE_PERSISTENCE_DATA_PATH,
                binary_path=WEAVIATE_BINARY_PATH
            ), 
            additional_headers={ 'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]})


        # self.chroma_client = chromadb.EphemeralClient()
        docs_vector_store = WeaviateVectorStore(weaviate_client = client, index_name="Docs", text_key="text")
        storage_context = StorageContext.from_defaults(vector_store=docs_vector_store )

        self.docs_index = VectorStoreIndex.from_vector_store(
            vector_store=docs_vector_store,
            storage_context = storage_context,
            service_context=ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
            )

        nodes_vector_store = WeaviateVectorStore(weaviate_client = client, index_name="Nodes", text_key="text")
        storage_context = StorageContext.from_defaults(vector_store=nodes_vector_store)
        self.nodes_index = VectorStoreIndex.from_vector_store(
            vector_store=nodes_vector_store,
            storage_context = storage_context,
            service_context=ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
            )


    def add(self, source: Union[Path, str]) -> bool:
        # Load configuration
        config = toml.load("../config.toml")
        data_dir = config["paths"]["DATA"]
        os.makedirs(data_dir, exist_ok=True)

        file_path= self._get_file(source, data_dir)

        # Read the document using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_path], file_extractor=CUSTOM_FILE_READER_CLS)
        new_documents = reader.load_data()
        print('New documents count:', len(new_documents))

        if (self._document_exists_in_library(file_path)):
            return False

        # Update document level metadata
        to_add_to_docs_index : List[BaseNode] = []
        for doc in new_documents:
            print('calling get_citation')
            citation = get_citation(doc)
            doc.metadata['citation']=citation
            to_add_to_docs_index.append(TextNode(text=citation, extra_info=doc.metadata))
            # setup docname?
        self.documents.extend(new_documents)
        self.docs_index.insert_nodes(to_add_to_docs_index)

        # Extract nodes : chunks and add node level metadata
        nodes = self._extract_nodes(new_documents)
        self.nodes_index.insert_nodes(nodes)
        # self.nodes.extend(nodes)

        return True
    
    def _document_exists_in_library(self, file_path):
        file_name = Path(file_path).name
        existing_file_names = [doc.metadata["file_name"] for doc in self.documents]
        if file_name in existing_file_names:
            return True
        return False

    def _get_file(self, source: Union[Path, str], data_dir: str) -> str:
        validated_source = SourceType(source=source).source  # Validate source using SourceType
        if urlparse(validated_source).scheme in ('http', 'https'):
            # Source is a URL, download the file and save it locally
            local_file_name = os.path.join(data_dir, os.path.basename(urlparse(validated_source).path))
            with requests.get(validated_source, stream=True) as r:
                r.raise_for_status()
                with open(local_file_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_file_name
        else:
            # Source is a local file, use it directly
            return str(validated_source)

    def _extract_nodes(self, documents: List[Document]) -> Sequence[BaseNode]:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=1024, chunk_overlap=256), #SentenceSplitter(chunk_size=512, chunk_overlap=128),
                # TitleExtractor(nodes=5),
                # QuestionsAnsweredExtractor(questions=3),
                # SummaryExtractor(summaries=["self"]),
                # KeywordExtractor(keywords=10),
            ]
        )
        nodes = pipeline.run(documents=documents)
        return nodes
    
    async def aquery(
            self,
            query: str,
            k: int = 10, 
            max_sources: int = 5,
            answer_length_prompt: str = "about 100 words",
            marginal_relevance: bool = True,
            answer: Optional[Answer] = None,
            doc_filter: Optional[bool] = None,
        ) -> Answer:
        """
        Answer a query.

        Args:
            query (str): The question to answer
            k (int, optional): The number of documents to retrieve. Defaults to 10.
            max_sources (int, optional): How much out of k documents are going to contribute to formulating the final answer. Defaults to 5.
            length_prompt (str, optional): The prompt for the answer length. Defaults to "about 100 words".
            marginal_relevance (bool, optional): Whether to use marginal relevance. Defaults to True.
            answer (Optional[Answer], optional): The Answer object to update. Defaults to None.
            doc_filter (Optional[bool], optional): Whether to filter documents based on their keys. Defaults to None.

        Returns:
            Answer: The updated Answer object
        """
        if answer is None:
            answer = Answer(query=query, answer_length=answer_length_prompt)

        if len(answer.contexts) == 0:
            # If we have more documents than the allowed number of documents to retrieve (k)
            # then we have to filter out documents

            #TODO: decide on re-ranking inside the adoc_match function ?
            answer.selected_documents = await self._adoc_match(answer.query)

            answer = await self._aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance
            )

            answer = await self._aget_answer(
                answer
            )
        

        return answer
    

    async def _aget_answer(self, answer: Answer) -> Answer:
        
        text_qa_template = answer_prompt.partial_format(answer_length=answer.answer_length, context=answer.context, query=answer.query)

        # response_synthesizer = get_response_synthesizer(
        #     response_mode="refine",
        #     service_context=self.service_context,
        #     text_qa_template=answer_prompt,
        #     refine_template=None,
        #     use_async=False,
        #     streaming=False,
        # )

        # # asynchronous
        # response = await response_synthesizer.asynthesize(
        #     answer.query,
        #     nodes=answer.contexts,
        #     additional_source_nodes=None,
        # )

        answer_text = self.service_context.llm.predict(prompt=text_qa_template)

        #TODO: formatted answer_text
        #TODO: Bibliography

        answer.answer = answer_text

        return answer


    async def _adoc_match(
        self,
        query: str,
        k: int = 25,
        rerank: Optional[bool] = None,
        # get_callbacks: CallbackFactory = lambda x: None,
    ) -> List[NodeWithScore]:

        # text_qa_template: Optional[BasePromptTemplate] = None,
        # refine_template: Optional[BasePromptTemplate] = None,
        # summary_template: Optional[BasePromptTemplate] = None,
        # simple_template: Optional[BasePromptTemplate] = None,

        # configure retriever (https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html)

        retriever = self.docs_index.as_retriever(
            similarity_top_k=k, 
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)])
        retrieved_nodes = retriever.retrieve(query)
        if retrieved_nodes is None or len(retrieved_nodes) == 0:
            return []
        
        # llm rerank for post procesing? https://docs.llamaindex.ai/en/stable/api_reference/node_postprocessor.html#llama_index.indices.postprocessor.LLMRerank

        return retrieved_nodes
    
    async def _aget_evidence(
        self,
        answer: Answer,
        k: int = 10,  # Number of vectors to retrieve
        max_sources: int = 5,  # Number of scored contexts to use
        marginal_relevance: bool = True,
        detailed_citations: bool = True,
        summarization: bool = True,
    ) -> Answer:

        if self.docs_index is None:
            return answer
        
        # NOTE we assume that the nodes index has been already built

        # docs = [ node_with_score.node for node_with_score in answer.selected_documents]
        # nodes = self._extract_nodes(docs)
        # self.nodes_index.insert_nodes(nodes)

        if self.nodes_index is None:
            return answer
        
        # Setting up the retriever
        # -----------------------
        # Example use of filters: https://docs.llamaindex.ai/en/stable/examples/vector_stores/pinecone_metadata_filter.html
        filters = []
        for selected_doc in answer.selected_documents:
            filters.append(MetadataFilter(key="citation", value=selected_doc.node.metadata["citation"]))

        metadata_filters = None
        if len(filters) > 0:
            metadata_filters = MetadataFilters(
                filters=filters,
                condition=FilterCondition.OR,
            ) 

        # add node postprocesor to create a summary for each text chunk relevant to the question
            
        node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.7)]
        if summarization == True:
            node_postprocessors.append(
                SummaryAndScoreNodePostProcessor(service_context=self.service_context)
            )

        nodes_retriever = self.nodes_index.as_retriever(
            similarity_top_k=k, 
            filters = metadata_filters,
            node_postprocessors=node_postprocessors
        )

        retrieved_nodes = nodes_retriever.retrieve(answer.query)

        answer.contexts.extend(retrieved_nodes)

        # Sort answer contexts by score (the score given by the summmarization prompt)
        # cut the contexts down to max_sources
            
        answer.contexts = sorted(
            retrieved_nodes,
            key=lambda x: x.get_score(),
            reverse=True,
        )[:max_sources]

        # Create context string
        #----------------------

        answer.context = self._get_context_str(answer.contexts, ask_llm=True, query=answer.query)    

        return answer
    
    @staticmethod
    def _find_metadata_summary_key(node: BaseNode) -> Optional[str]:
        for key in node.metadata.keys():
            if'summary' in key:
                return key
        return None
    
    def _get_context_str(self, contexts: List[NodeWithScore], detailed_citations=True, ask_llm = True, query = '') -> str:
        #TODO Review the concept of text.name ??? here using node._id
        context_str = "\n\n".join(
        [
            f"{node.node.id_}: {node.node.text}" +
            (f"\n\nBased on {node.node.metadata['citation']}" if detailed_citations else '' )
                for node in contexts
        ])

        valid_names = [node.id_ for node in contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)
        if ask_llm == True and query != '':
            extra_background_information=f'\n\nExtra background information:{self._ask_llm(query)}'
        return context_str
    
    def _ask_llm(self, query: str)-> str:
        prompt_tpl = ask_llm_prompt.partial_format(query=query) 

        answer = self.service_context.llm.predict(prompt=prompt_tpl)

        return answer
