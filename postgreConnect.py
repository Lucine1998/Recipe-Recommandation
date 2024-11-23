import psycopg2
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# To create structured vectorstore via PostgreDB

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


def create_recipes_embeddings_table(batch_size=100):
    try:
        # Establish a database connection
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # Ensure the pgvector extension is installed
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("pgvector extension ensured available.")

        # Create a new table `recipes_embeddings`
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes_embeddings (
            id BIGINT PRIMARY KEY,
            embedding VECTOR(768)  -- Vector type for pgvector
        );
        """)
        print("Table `recipes_embeddings` created successfully or already exists.")

        # Query data from the original table to process
        cursor.execute("SELECT id, description FROM recipes;")
        rows = cursor.fetchall()

        # Check which embeddings already exist
        cursor.execute("SELECT id FROM recipes_embeddings;")
        existing_ids = {row[0] for row in cursor.fetchall()}

        # Prepare data for processing
        to_process = [(row[0], row[1]) for row in rows if row[0] not in existing_ids]
        print(f"Number of records to process embeddings: {len(to_process)}")

        # Generate embeddings in batches and insert into the database
        batch = []
        for record_id, description in tqdm(to_process, desc="Generating embeddings"):
            if description:  # Ensure the description is not empty
                embedding = model.encode(description).tolist()
                batch.append((record_id, embedding))

                # Insert every `batch_size` records
                if len(batch) >= batch_size:
                    cursor.executemany("""
                    INSERT INTO recipes_embeddings (id, embedding)
                    VALUES (%s, %s);
                    """, batch)
                    connection.commit()
                    batch = []

        # Insert the remaining records
        if batch:
            cursor.executemany("""
            INSERT INTO recipes_embeddings (id, embedding)
            VALUES (%s, %s);
            """, batch)
            connection.commit()

        print("Embedding generation and storage completed!")

    except Exception as error:
        print("Operation failed. Error details:", error)

    finally:
        # Close the database connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Database connection closed.")


def similarity_search(query, top_k=5):
    try:
        # Establish a database connection
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

        # Perform similarity search
        cursor.execute("""
        SELECT id, embedding <=> %s AS similarity
        FROM recipes_embeddings
        ORDER BY similarity
        LIMIT %s;
        """, (query_embedding, top_k))
        results = cursor.fetchall()

        print("Similarity search results:")
        for result in results:
            print(f"ID: {result[0]}, Similarity: {result[1]}")

    except Exception as error:
        print("Similarity search failed. Error details:", error)

    finally:
        # Close the database connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Database connection closed.")


# Run the script
if __name__ == "__main__":
    create_recipes_embeddings_table(batch_size=100)

    # Example query
    query_text = "We have blueberry and honey at home, can you recommend us some recipes to make full use of our food at home."
    similarity_search(query_text, top_k=5)
