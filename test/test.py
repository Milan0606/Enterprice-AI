
import os
import json
import boto3

from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

region = os.getenv("AWS_REGION")

client = boto3.client(
    "bedrock-runtime",
    region_name=region,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

print("AWS Region:", region)


# Test 1: Amazon Titan Embeddings
try:
    print("\nTesting Titan Embeddings...")

    embedding_response = client.invoke_model(
        modelId=os.getenv("BEDROCK_EMBED_MODEL"),
        body=json.dumps({
            "inputText": "Hello Amazon Bedrock",
            "dimensions": 1024,
            "normalize": True
        }),
        contentType="application/json",
        accept="application/json"
    )

    embedding_data = json.loads(
        embedding_response["body"].read()
    )

    print(
        "Embedding dimension:",
        len(embedding_data["embedding"])
    )

except ClientError as e:
    print("Titan Embedding Error:")
    print("Error Code:", e.response["Error"]["Code"])
    print("Message:", e.response["Error"]["Message"])


# Test 2: Amazon Nova Lite
try:
    print("\nTesting Nova Lite...")

    chat_response = client.converse(
        modelId=os.getenv("BEDROCK_CHAT_MODEL"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Explain RAG in one sentence."}
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 200,
            "temperature": 0.3
        }
    )

    print(
        "Nova Response:",
        chat_response["output"]["message"]["content"][0]["text"]
    )

except ClientError as e:
    print("Nova Lite Error:")
    print("Error Code:", e.response["Error"]["Code"])
    print("Message:", e.response["Error"]["Message"])
