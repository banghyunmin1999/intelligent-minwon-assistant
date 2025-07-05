from setuptools import setup, find_packages

setup(
    name="intelligent-minwon-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "pydantic-settings",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-huggingface",
        "langchain-postgres",
        "psycopg[binary]",
        "asyncpg",
        "sqlalchemy",
        "PyPDF2",
        "pgvector",
        "rank_bm25",
        "torch",
        "numpy",
        "sentence-transformers",
    ],
)
