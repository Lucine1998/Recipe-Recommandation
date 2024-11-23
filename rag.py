import psycopg2
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from llm_client_scaleway import LLMClient


# Load environment variables
load_dotenv()

# Database connection parameters
host = os.getenv("SCW_DB_HOST")
port = os.getenv("SCW_DB_PORT")
database = os.getenv("SCW_DB_NAME")
user = os.getenv("SCW_DB_USER")
password = os.getenv("SCW_DB_PASSWORD")

# Initialize the SentenceTransformer model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Initialize LLMClient
client = LLMClient(
    api_url="https://api.scaleway.ai/e63882dd-2049-4317-a9ee-d03fc21c4ca8/v1/chat/completions",
    api_key=os.getenv("SCW_SECRET_KEY")
)


def similarity_search(query, top_k=5):
    """
    Perform similarity search on the recipes_embeddings table based on the query,
    and return the corresponding full content from the recipes table.
    """
    try:
        # Create a database connection
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # Generate the embedding for the query
        query_embedding = model.encode(query).tolist()

        # Convert the embedding into PostgreSQL array format
        query_embedding_str = f"ARRAY{query_embedding}"
        # Use SQL parameterization to avoid syntax issues
        sql_query = f"""
        SELECT r.*
        FROM recipes_embeddings e
        JOIN recipes r ON e.id = r.id
        ORDER BY e.embedding <=> %s::VECTOR
        LIMIT %s;
        """
        cursor.execute(sql_query, (query_embedding, top_k))
        results = cursor.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cursor.description]

        # Format results into a readable context
        formatted_results = []
        for row in results:
            record = {column: value for column, value in zip(column_names, row)}
            formatted_results.append(record)

        return formatted_results

    except Exception as error:
        print("Similarity search failed, error details:", error)
        return []

    finally:
        # Close the connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def ask_question_with_context(question, context):
    """
    Ask LLM a question with a given context using LLMClient.
    """
    # Combine the question with the context
    context_text = "\n".join([f"- {item}" for item in context])
    full_message = (
        f"Here are some relevant recipes based on your input:\n"
        f"{context_text}\n\n"
        f"Question: {question}"

        "Keep the response concise and no numeric info included in answer (expect time)."
        "Make sure to mention if some ingredients in the recipe we dont have at home."
    )

    print(f"Question: {question}")
    # print(f"Asking LLMClient with message:\n{full_message}")
    response = client.generate_response(user_message=full_message, stream=False)
    print("Response:", response)
    return response


if __name__ == "__main__":
    # Example query
    query_text = "I have 2 eggs at home. Can you recommend recipes that use these ingredients effectively?"

    # Retrieve top 10 similar recipes
    search_results = similarity_search(query_text, top_k=10)

    # Format results into strings for LLM context
    formatted_context = [
        f"Name: {res['name']}, Description: {res['description']}, Preparation time (minutes): {res['minutes']}, Nutrition (calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat): {res['nutrition']}"
        for res in search_results
    ]

    # Use formatted context directly for querying
    # question = "Based on the available ingredients, suggest the top 3 most nutritious recipes."
    question = "Based on the available ingredients, suggest the top 3 eastest recipes to make."
    ask_question_with_context(question, formatted_context)