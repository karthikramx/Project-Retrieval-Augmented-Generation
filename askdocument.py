'''
Question Answering From Various Sources of Text
# Wikipedia
# Websites
# PDFs
# Documents
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_pinecone import Pinecone as langPineCone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
import warnings
import tiktoken
import os

warnings.filterwarnings('ignore')

class AskDocument():
    def __init__(self) -> None:
        '''
        
        '''
        load_dotenv(find_dotenv(), override=True)
        self.pinecone = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        self.embeddings = OpenAIEmbeddings()
        self.chunks = None
        self.vector_store = None

    ## TODO
    def load_from_wikipedia(sefl,query, lang='en', load_max_docs=2):
        loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
        data = loader.load()
        return data

    def chunk_data(self,data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
        self.chunks = text_splitter.split_documents(data)

    def print_embedding_cost(self,texts):
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum(len(enc.encode(page.page_content)) for page in texts)
        print(f'Total Tokens:{total_tokens}')
        print(f'Embedding cost in USD:{total_tokens/1000*0.0004:.6f}')

    def delete_pinecone_index(self,index_name='all'):
        if index_name == 'all':
            indexes = self.pinecone.list_indexes()
            print("Deleting all indexes")
            for index in indexes:
                self.pinecone.delete_index(index['name'])
            print("Done deleting all indexes")
        else:
            print(f"Deleting index: {index_name} ...", end='')
            self.pinecone.delete_index(index_name)
            print(f"Done deleting {index_name}")


    def insert_or_fetch_embeddings(self, index_name):
        list_of_indexes = [index['name'] for index in self.pinecone.list_indexes()]

        if index_name in list_of_indexes:
            print(f'Index {index_name} already exists. Loading embeddings ...',end='')
            self.vector_store = langPineCone.from_existing_index(index_name=index_name, embedding=self.embeddings)
        else:
            self.chunk_data(self.load_from_wikipedia(index_name))
            self.print_embedding_cost(self.chunks)
            print(f'Creating index {index_name} and embedddings ...', end='')
            self.pinecone.create_index(index_name, 
                                    dimension=1536, 
                                    metric='cosine',
                                    spec= {'serverless': {'cloud': 'aws', 'region': 'us-west-2'}})
            self.vector_store = langPineCone.from_documents(self.chunks, 
                                                        self.embeddings, 
                                                        index_name=index_name,
                                                        )

    def ask_and_get_answer(self, q):
        llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=1)
        retriever = self.vector_store.as_retriever(search_type='similarity',
                                            search_kwargs={'k':3})
        
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        print(chain)
        answer = chain.invoke(q)
        print(answer)
        print("******")
        resp = retriever.get_relevant_documents(q)
        for r in resp:
            print("-------")
            print(r.page_content)
        return answer
    

    def get_list_of_indexes(self):
        data = self.pinecone.list_indexes()
        # return a list of names
        return [index['name'] for index in data]
    
    def set_topic(self, topic):
        self.insert_or_fetch_embeddings(index_name=topic)   


if __name__ == '__main__':
    AD = AskDocument()
    print(AD.get_list_of_indexes())

    topic = input(f"Enter a topic you want to explore")
    data = AD.load_from_wikipedia(topic)
    AD.chunk_data(data)
    AD.insert_or_fetch_embeddings(index_name=topic)

    while True:
        question = input(f"\nEnter your question about {topic}, enter q to exit:")
        if len(question) == 0: break
        answer = AD.ask_and_get_answer(q=question)
        print("\nAnswer:",answer["result"])
        print("\n"+"-"*50)

    AD.delete_pinecone_index()

    
