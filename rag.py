import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to PostgreDB
host = os.getenv("SCW_DB_HOST")  # Database server address, e.g., "127.0.0.1" or the server's IP address
port = os.getenv("SCW_DB_PORT")  # Database port number, PostgreSQL default is 5432
database = os.getenv("SCW_DB_NAME")  # Database name
user = os.getenv("SCW_DB_USER")  # Database username
password = os.getenv("SCW_DB_PASSWORD")  # Database password

try:
    # Establish the connection
    connection = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    print("Connection successful!")

    # Create a cursor
    cursor = connection.cursor()

    # Execute a simple SQL query
    cursor.execute("SELECT version();")  # Query PostgreSQL version
    version = cursor.fetchone()
    print("PostgreSQL version:", version)

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as error:
    print("Connection failed, error message:", error)

