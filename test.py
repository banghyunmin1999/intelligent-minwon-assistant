import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

print("os.environ['DB_HOST']:", os.environ.get('DB_HOST'))
print("os.getenv('DB_HOST'):", os.getenv('DB_HOST'))