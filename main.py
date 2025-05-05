
# import os
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from neo4j import GraphDatabase
# from tika import parser
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from tqdm import tqdm
# import spacy
# from huggingface_hub import InferenceClient
# from pydantic import BaseModel
# import re
# import logging

# load_dotenv()

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# try:
#     uri = os.environ["NEO4J_URI"]
#     driver = GraphDatabase.driver(uri, auth=None)
# except KeyError:
#     logging.error("NEO4J_URI environment variable not set.")
#     exit(1)

# nlp = spacy.load("en_core_web_sm")

# huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# if not huggingfacehub_api_token:
#     raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token

# # model_name = "google/t5-small"

# model_name = "google/flan-t5-large"
# client = InferenceClient(model=model_name, token=huggingfacehub_api_token)

# def generate_cypher_huggingface_api(natural_language_query: str):
#     prompt = f"""
#     You are an expert in converting English questions to Neo4j Cypher Graph code! Your task is to generate Cypher queries to retrieve data from a Neo4j graph database. The graph database contains nodes representing uploaded documents, specifically 'Chunk' nodes.
#     The 'Chunk' node label has the following property: 'content'. This property stores the textual content of the document chunks.

#     Your goal is to generate Cypher queries that accurately retrieve information from these 'Chunk' nodes based on the user's natural language questions.

#     Rules:

#     1. **Focus on 'Chunk' Nodes:** All queries should primarily target the 'Chunk' nodes.
#     2. **Search within 'content':** Use the 'content' property of the 'Chunk' nodes to search for information.
#     3. **Directly Executable Cypher:** Generate Cypher queries that are directly executable on a Neo4j database.
#     4. **Case-Insensitive Searching:** Use the `toLower()` function for case-insensitive searches.
#     5. **Use `CONTAINS` for substring matching:** For finding text within the `c.content` property, always use the `CONTAINS` function.
#     6. **Always include a `RETURN` clause:** Specify what data to retrieve from the 'Chunk' nodes.
#     7. **Specifically return the `c.content` property:** Only return the content of the chunks.
#     8. **No extra information:** Do not add any extra information to the output, other than the Cypher query.
#     9. **Ensure Executable Cypher:** Make absolutely certain that the generated Cypher query is valid and executable.
#     10. **Avoid `NOT IN` for string matching:** Do not use `NOT IN` for searching within the `c.content` property. It's not appropriate for string matching.
#     11. **Use `CONTAINS` for finding substrings:** Specifically, when the user asks to "find about" or "find" something, translate it to a `CONTAINS` clause in your Cypher query.
#     12. **Handle phrases correctly:** If the user query includes a phrase like 'Future of AI', ensure the Cypher query searches for that exact phrase using `CONTAINS`.
#     13. **Handle "what is" questions:** If the user asks "what is" something, translate it to a `CONTAINS` clause to find the definition or explanation in the content.

#     Examples:

#     Natural Language: Find chunks containing 'Machine Learning'
#     Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('Machine Learning') RETURN c.content

#     Natural Language: What is 'Artificial Intlelligence'?
#     Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('Artificial Intelligence') RETURN c.content

#     Natural Language: Show me all chunks about 'data analysis'
#     Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('data analysis') RETURN c.content

#     Natural Language: What is the 'future of AI'?
#     Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('future of AI') RETURN c.content

#     Natural Language: {natural_language_query}
#     Cypher:
#     """
#     try:
#         response = client.text_generation(prompt, max_new_tokens=200)
#         logging.debug(f"Hugging Face response: {response}")
#         return response
#     except Exception as e:
#         if "402 Client Error: Payment Required" in str(e):
#             return "Hugging Face API credit limit exceeded. Please subscribe to PRO or consider local inference."
#         else:
#             logging.error(f"Error generating Cypher: {e}")
#             return f"Error generating Cypher: {e}"

# def extract_cypher_query(generated_text: str) -> str:
#     """Extracts the Cypher query from the generated text."""
#     cypher_match = re.search(r"(MATCH|RETURN|WHERE|CREATE|DELETE).*$", generated_text, re.DOTALL)
#     if cypher_match:
#         cypher_query = cypher_match.group(0).strip().rstrip(';')
#         logging.debug(f"Extracted Cypher query: {cypher_query}") # Debugging Step 2

#         return cypher_query
#     else:
#         logging.error(f"Could not extract Cypher query. Generated text: {generated_text}")
#         return ""

# def is_valid_cypher(cypher_query: str) -> bool:
#     """Validates if the generated Cypher query is valid."""
#     try:
#         with driver.session() as session:
#             session.run(cypher_query, limit=0)  # Try running the query with limit 0
#         return True
#     except Exception:
#         logging.debug(f"Cypher validation error: {e}") # Debugging Step 3
#         return False

# def execute_cypher(cypher_query):
#     try:
#         logging.debug(f"Executing Cypher query: {cypher_query}") # Debugging Step 1
#         if not is_valid_cypher(cypher_query):
#             logging.error("Cypher query is not valid.") # Debugging Step 2
#             return "Invalid Cypher query generated."

#         with driver.session() as session:
#             logging.debug("Session created.") # Debugging Step 3
#             result = session.run(cypher_query)
#             logging.debug("Query executed.") # Debugging Step 4
#             data = [record["c.content"] for record in result]  # Extract only the content
#             logging.info(f"Neo4j Results: {data}")
#             return data
#     except Exception as e:
#         logging.error(f"Error executing Cypher query: {e}")
#         return f"Error executing Cypher query: {e}"

# def upload_chunk_and_entities_to_neo4j(file_name, chunk_content, chunk_index):
#     try:
#         doc = nlp(chunk_content)
#         with driver.session() as session:
#             session.execute_write(
#                 lambda tx, name, content, index: tx.run(
#                     "MERGE (c:Chunk {fileName: $name, content: $content, chunkIndex: $index})",
#                     name=name,
#                     content=content,
#                     index=index,
#                 ),
#                 file_name,
#                 chunk_content,
#                 chunk_index,
#             )

#             for ent in doc.ents:
#                 session.execute_write(
#                     lambda tx, text, label, chunk_index: tx.run(
#                         "MERGE (e:Entity {text: $text, label: $label}) "
#                         "MERGE (c:Chunk {chunkIndex: $chunk_index}) "
#                         "MERGE (c)-[:CONTAINS_ENTITY]->(e)",
#                         text=ent.text,
#                         label=ent.label_,
#                         chunk_index=chunk_index,
#                     ),
#                     ent.text,
#                     ent.label_,
#                     chunk_index,
#                 )
#     except Exception as e:
#         logging.error(f"Error processing chunk: {e}")

# def extract_text_from_file(file_content):
#     try:
#         parsed = parser.from_buffer(file_content)
#         if parsed and parsed["content"]:
#             return parsed["content"].encode('utf-8', errors='replace').decode('utf-8')
#         else:
#             return None
#     except Exception as e:
#         logging.error(f"Error extracting text: {e}")
#         return None

# def process_and_upload_chunks(file_filename, extracted_content):
#     """Processes content, splits it into smaller chunks, and uploads to Neo4j."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=250,  # Reduced chunk size to 250
#         chunk_overlap=50,  # Reduced overlap to 50
#         length_function=len,
#         separators=["\n\n", "\n", " ", "."],  # Split by double newlines, newlines, spaces, and periods
#     )
#     documents = text_splitter.create_documents([extracted_content])

#     for i, d in tqdm(enumerate(documents), total=len(documents)):
#         chunk_text = d.page_content.strip()
#         if chunk_text and chunk_text[-1] != '.':  # Ensure chunk ends with a period
#             sentences = chunk_text.split('. ')
#             if len(sentences) > 1:
#                 chunk_text = '. '.join(sentences[:-1]) + '.'
#             else:
#                 chunk_text += '.'
#         if chunk_text:  # Only upload non-empty chunks
#             upload_chunk_and_entities_to_neo4j(file_filename, chunk_text, i)
#             logging.info(f"Chunk {i} uploaded to Neo4j.")
#     logging.info("File uploaded and processed successfully to Neo4j!")

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     try:
#         if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
#             raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, TXT, and DOCX files are allowed.")

#         file_content = await file.read()
#         extracted_content = extract_text_from_file(file_content)

#         if extracted_content is None:
#             raise HTTPException(status_code=400, detail="Could not extract text from file.")

#         process_and_upload_chunks(file.filename, extracted_content)

#         return {"filename": file.filename, "message": "File uploaded and processed successfully to Neo4j!"}

#     except HTTPException as http_exception:
#         raise http_exception
#     except Exception as e:
#         logging.error(f"Error during file upload: {e}")
#         raise HTTPException(status_code=500, detail=f"Error: {e}")

# class QueryRequest(BaseModel):
#     natural_language_query: str


# @app.post("/query/")
# async def query_neo4j(query_request: QueryRequest):
#     """Queries Neo4j using Hugging Face Inference API."""
#     try:
#         cypher_query = generate_cypher_huggingface_api(query_request.natural_language_query)
#         extracted_cypher_query = extract_cypher_query(cypher_query)

#         if extracted_cypher_query:
#             logging.info(f"Generated Cypher Query: {extracted_cypher_query}")
#             neo4j_results = execute_cypher(extracted_cypher_query)
#             return {"result": neo4j_results}
#         else:
#             return {"error": "Could not extract Cypher query from generated text."}

#     except Exception as e:
#         logging.error(f"Error querying Neo4j: {e}")
#         return {"error": f"Error querying Neo4j: {e}"} # ensures that error key is always present.

# @app.on_event("shutdown")
# def shutdown_event():
#     """Closes the Neo4j driver on shutdown."""
#     if driver:
#         driver.close()


import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from neo4j import GraphDatabase
from tika import parser
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import spacy
from pydantic import BaseModel
import re
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    uri = os.environ["NEO4J_URI"]
    driver = GraphDatabase.driver(uri, auth=None)
except KeyError:
    logging.error("NEO4J_URI environment variable not set.")
    exit(1)

nlp = spacy.load("en_core_web_sm")

# Local Model Setup
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, num_beams=4)

def generate_cypher_local(natural_language_query: str):
    prompt = f"""
    You are an expert in converting English questions to Neo4j Cypher Graph code! Your task is to generate Cypher queries to retrieve data from a Neo4j graph database. The graph database contains nodes representing uploaded documents, specifically 'Chunk' nodes.
    The 'Chunk' node label has the following property: 'content'. This property stores the textual content of the document chunks.

    Your goal is to generate Cypher queries that accurately retrieve information from these 'Chunk' nodes based on the user's natural language questions.

    Rules:


        1. **Focus on 'Chunk' Nodes:** All queries should primarily target the 'Chunk' nodes.
        2. **Search within 'content':** Use the 'content' property of the 'Chunk' nodes to search for information.
        3. **Directly Executable Cypher:** Generate Cypher queries that are directly executable on a Neo4j database.
        4. **Case-Insensitive Searching:** Use the `toLower()` function for case-insensitive searches.
        5. **Use `CONTAINS` for substring matching:** For finding text within the `c.content` property, always use the `CONTAINS` function.
        6. **Always include a `MATCH` clause:** Start the query with a `MATCH (c:Chunk)` clause.
        7. **Ensure Executable Cypher:** Make absolutely certain that the generated Cypher query is valid and executable.
        8. **Avoid `NOT IN` for string matching:** Do not use `NOT IN` for searching within the `c.content` property. It's not appropriate for string matching.
        9. **Use `CONTAINS` for finding substrings:** Specifically, when the user asks to "find about", "find" or "what is" something, translate it to a `CONTAINS` clause in your Cypher query.
        10. **Handle phrases correctly:** If the user query includes a phrase like 'Future of AI', ensure the Cypher query searches for that exact phrase using `CONTAINS`.
        11. **Handle "what is" questions:** If the user asks "what is" something, translate it to a `CONTAINS` clause to find the definition or explanation in the content.


    Examples:
    
    Natural Language: what is the future of blockchain
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('future of blockchain') RETURN c.content


    Natural Language: Find about 'Machine Learning'
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('Machine Learning') RETURN c.content

    Natural Language: What is abstract?
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('abstract') RETURN c.content

    Natural Language: what is the title of document
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('title') RETURN c.content

    Natural Language: who is the author
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('author') RETURN c.content

    Natural Language: Find the conclusion
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('conclusion') RETURN c.content

    Natural Language: date of publication
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('date') RETURN c.content

    Natural Language: what are the challenges
    Cypher: MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower('challenges') RETURN c.content

    Natural Language: document name or file name
    Cypher: MATCH (c:Chunk) RETURN c.fileName



 
    Natural Language: {natural_language_query}
    Cypher:
    """
    try:
        response = generator(prompt, max_length=512)[0]['generated_text']
        logging.debug(f"Local Model Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error generating Cypher: {e}")
        return f"Error generating Cypher: {e}"

def extract_cypher_query(generated_text: str) -> str:
    """Extracts the Cypher query from the generated text."""
    cypher_match = re.search(r"(MATCH|RETURN|WHERE|CREATE|DELETE).*$", generated_text, re.DOTALL)
    if cypher_match:
        cypher_query = cypher_match.group(0).strip().rstrip(';')
        if "RETURN c.fileName" not in cypher_query and "RETURN c.content" not in cypher_query: # Ensure a return is present
            cypher_query += " RETURN c.content"  # Default return
            logging.debug(f"Extracted Cypher query: {cypher_query}")
        return cypher_query
    else:
        logging.error(f"Could not extract Cypher query. Generated text: {generated_text}")
        logging.debug(f"Full generated Text: {generated_text}")
        return ""

def is_valid_cypher(cypher_query: str) -> bool:
    """Validates if the generated Cypher query is valid."""
    try:
        with driver.session() as session:
            session.run(cypher_query, limit=0)
        return True
    except Exception as e:
        logging.debug(f"Cypher validation error: {e}, for query: {cypher_query}") #Enhanced logging
        return False

def execute_cypher(cypher_query: str) -> list | dict | str:
    """Executes the Cypher query and returns the results."""
    try:
        if not is_valid_cypher(cypher_query):
            return ["Invalid Cypher query generated."]

        with driver.session() as session:
            result = session.run(cypher_query + " LIMIT 100")  # Limit results
            if "RETURN c.fileName" in cypher_query:
                data = [record["c.fileName"] for record in result]
                return data
            elif "RETURN e.text, e.label" in cypher_query:
                data = [{"text": record["e.text"], "label": record["e.label"]} for record in result]
                return data
            elif "RETURN c.content" in cypher_query:
                data = [record["c.content"] for record in result]
                return data
                if not data:
                    return ["No matching content found."]
                return data
            else:
                return ["No data returned"] # added to handle empty returns
    except Exception as e:
        logging.error(f"Error executing Cypher query: {e}. Query: {cypher_query}")
        return [f"Error executing Cypher query: {e}"]


def upload_chunk_and_entities_to_neo4j(file_name, chunk_content, chunk_index):
    try:
        doc = nlp(chunk_content)
        with driver.session() as session:
            def create_chunk_and_entities(tx, name, content, index, entities):
                tx.run(
                    "MERGE (c:Chunk {fileName: $name, content: $content, chunkIndex: $index})",
                    name=name,
                    content=content,
                    index=index,
                )
                for ent in entities:
                    tx.run(
                        "MERGE (e:Entity {text: $text, label: $label}) "
                        "MERGE (c:Chunk {chunkIndex: $chunk_index}) "
                        "MERGE (c)-[:CONTAINS_ENTITY]->(e)",
                        text=ent.text,
                        label=ent.label_,
                        chunk_index=index,
                    )

            session.execute_write(create_chunk_and_entities, file_name, chunk_content, chunk_index, doc.ents)

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")



def extract_text_from_file(file_content):
    try:
        parsed = parser.from_buffer(file_content)
        if parsed and parsed["content"]:
            return parsed["content"].encode('utf-8', errors='replace').decode('utf-8')
        else:
            return None
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return None

def process_and_upload_chunks(file_filename, extracted_content):
    """Processes content, splits it into smaller chunks, and uploads to Neo4j."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size to 250
        chunk_overlap=80,  # Reduced overlap to 50
        length_function=len,
        separators=["\n\n", "\n", " ", "."],  # Split by double newlines, newlines, spaces, and periods
    )
    documents = text_splitter.create_documents([extracted_content])

    for i, d in tqdm(enumerate(documents), total=len(documents)):
        chunk_text = d.page_content.strip()
        if chunk_text and not chunk_text.endswith(('.', '?', '!')):
            sentences = chunk_text.split('. ')
            if len(sentences) > 1:
                chunk_text = '. '.join(sentences[:-1]) + '.'
            else:
                chunk_text += '.'
        if chunk_text:
            upload_chunk_and_entities_to_neo4j(file_filename, chunk_text, i)
            logging.info(f"Chunk {i} uploaded to Neo4j.")
    logging.info("File uploaded and processed successfully to Neo4j!")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, TXT, and DOCX files are allowed.")

        file_content = await file.read()
        extracted_content = extract_text_from_file(file_content)

        if extracted_content is None:
            raise HTTPException(status_code=400, detail="Could not extract text from file.")

        process_and_upload_chunks(file.filename, extracted_content)

        return {"filename": file.filename, "message": "File uploaded and processed successfully to Neo4j!"}

    except HTTPException as http_exception:
        raise http_exception
    except Exception as e:
        logging.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

class QueryRequest(BaseModel):
    natural_language_query: str



@app.post("/query/")
async def query_neo4j(query_request: QueryRequest):
    """Queries Neo4j using Local Inference."""
    try:
        cypher_query = generate_cypher_local(query_request.natural_language_query)
        extracted_cypher_query = extract_cypher_query(cypher_query)

        if extracted_cypher_query:
            logging.info(f"Generated Cypher Query: {extracted_cypher_query}")
            print(f"Generated Cypher Query: {extracted_cypher_query}")  # Print to terminal

            neo4j_results = execute_cypher(extracted_cypher_query)

            # Attempt to extract direct answer using the local model
            if neo4j_results and isinstance(neo4j_results, list) and len(neo4j_results) > 0:
                context = " ".join(neo4j_results[:4])
                direct_answer_prompt = f"""
                    You are an expert assistant helping users understand information from documents.

                    Based on the following context extracted from documents, answer the user's question in a detailed, informative, and helpful manner. Provide multiple sentences if needed. If the user's question are specific then give the correct specific answer only.
                

                    Context:
                    {context}

                    Question:
                    {query_request.natural_language_query}

                    Direct Answer:
                """
                try:
                    direct_answer = generator(direct_answer_prompt, max_length=400, num_return_sequences=1)[0]['generated_text'].strip()

                    return {"result": direct_answer}
                except Exception as e:
                    logging.error(f"Error generating direct answer: {e}")
                    return {"result": neo4j_results} # Return the original results if direct answer generation fails
            else:
                return {"result": neo4j_results}

        else:
            return {"error": "Could not extract Cypher query from generated text."}

    except Exception as e:
        logging.error(f"Error querying Neo4j: {e}")
        return {"error": f"Error querying Neo4j: {e}"}


@app.on_event("shutdown")
def shutdown_event():
    """Closes the Neo4j driver on shutdown."""
    if driver:
        driver.close()