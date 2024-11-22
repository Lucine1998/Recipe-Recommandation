import psycopg2
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# 数据库连接参数
host = os.getenv("SCW_DB_HOST")
port = os.getenv("SCW_DB_PORT")
database = os.getenv("SCW_DB_NAME")
user = os.getenv("SCW_DB_USER")
password = os.getenv("SCW_DB_PASSWORD")

# 初始化 SentenceTransformer 模型
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def create_recipes_embeddings_table(batch_size=100):
    try:
        # 创建数据库连接
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # 确保 pgvector 扩展已安装
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("pgvector 扩展已确保可用。")

        # 创建新表 recipes_embeddings
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes_embeddings (
            id BIGINT PRIMARY KEY,
            embedding VECTOR(768)  -- pgvector 的向量类型
        );
        """)
        print("表 recipes_embeddings 创建成功或已存在。")

        # 查询原表中需要处理的数据
        cursor.execute("SELECT id, description FROM recipes;")
        rows = cursor.fetchall()

        # 检查哪些嵌入已经存在
        cursor.execute("SELECT id FROM recipes_embeddings;")
        existing_ids = {row[0] for row in cursor.fetchall()}

        # 准备数据
        to_process = [(row[0], row[1]) for row in rows if row[0] not in existing_ids]
        print(f"需生成嵌入的数据量: {len(to_process)}")

        # 批量生成嵌入并插入
        batch = []
        for record_id, description in tqdm(to_process, desc="生成嵌入"):
            if description:  # 确保 description 非空
                embedding = model.encode(description).tolist()
                batch.append((record_id, embedding))

                # 每 batch_size 插入一次
                if len(batch) >= batch_size:
                    cursor.executemany("""
                    INSERT INTO recipes_embeddings (id, embedding)
                    VALUES (%s, %s);
                    """, batch)
                    connection.commit()
                    batch = []

        # 插入最后一批数据
        if batch:
            cursor.executemany("""
            INSERT INTO recipes_embeddings (id, embedding)
            VALUES (%s, %s);
            """, batch)
            connection.commit()

        print("嵌入生成并存储完成！")

    except Exception as error:
        print("操作失败，错误信息:", error)

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("数据库连接已关闭。")


def similarity_search(query, top_k=5):
    try:
        # 创建数据库连接
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # 生成查询的嵌入
        query_embedding = model.encode(query).tolist()

        # 执行相似性搜索
        cursor.execute("""
        SELECT id, embedding <=> %s AS similarity
        FROM recipes_embeddings
        ORDER BY similarity
        LIMIT %s;
        """, (query_embedding, top_k))
        results = cursor.fetchall()

        print("相似性搜索结果：")
        for result in results:
            print(f"ID: {result[0]}, Similarity: {result[1]}")

    except Exception as error:
        print("相似性搜索失败，错误信息:", error)

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("数据库连接已关闭。")


# 运行脚本
if __name__ == "__main__":
    create_recipes_embeddings_table(batch_size=100)

    # 示例查询
    query_text = "We have blueberry and honey at home, can you recommend us some recipes to make fully use of our food at home."
    similarity_search(query_text, top_k=5)
