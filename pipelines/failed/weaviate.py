"""
title: Llama Index Pipeline
author: jellz77
date: 2024-06-03
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Langchain.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
#from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        self.client = None
        self.weaviate_client = None
        self.str = None
    async def on_startup(self):       
        
        from langchain_community.document_loaders import DirectoryLoader                         
        from langchain_community.vectorstores.weaviate import Weaviate
        from langchain_community.embeddings.ollama import OllamaEmbeddings
        import weaviate
        
        print("JS Debugging Started")
        
        self.client = weaviate.Client("http://localhost:8081")
        self.weaviate_client = weaviate.connect_to_local("localhost","8081")#v4
        index_name="Pipeline_test"    

        print("Connection to Weaviate successful")
        from weaviate.classes.config import Configure
        from weaviate.classes.config import Property, DataType

        self.weaviate_client.collections.delete_all()
        self.weaviate_client.collections.list_all()
        print("Collections successfully deleted")
        self.weaviate_client.collections.create(
            index_name,
            #see notes above re: the docker modules that need to be enabled for text2vec* to work correctly -e ENABLE_MODULES=text2vec-ollama
            vectorizer_config=Configure.Vectorizer.text2vec_ollama( 
                model="nomic-embed-text",    
                api_endpoint="http://host.docker.internal:11434",
            ),
            # generative_config=Configure.Generative.ollama(
            #     api_endpoint = "http://host.docker.internal:11434",
            #     model="jonphi"
            # ),

            # properties=[
            #     Property(name="page_content", data_type=DataType.TEXT),
            #     Property(name="source", data_type=DataType.INT),
            # ]

        )
        print(f"Collection {index_name} successfully created")        
        self.documents = DirectoryLoader("/home/jonot480/Documents/pipeline_docs_test").load()
        # self.index = WeaviateVectorStore.from_documents(
        #     self.client,
        #     self.documents
        # )
        for i in self.documents:
            print(f"Document ingestion loop - {i.metadata.get("source")} has been ingested!")
            #print(i.metadata.get("source"))

        self.index = Weaviate.from_documents(
            documents=self.documents, 
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
            client=self.client,    
            index_name=index_name,
            #prefer_grpc=True, 
        ) 
        print("Retriever is ready")
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        #print(messages)
        #print(user_message)

        
        #query_engine = self.index.as_query_engine(streaming=True)
        response = self.index.invoke(user_message)
        #strin = "hi im jon"
        #return strin
        return response.response_gen
    

