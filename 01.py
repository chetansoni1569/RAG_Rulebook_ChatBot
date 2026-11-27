
import fitz
import re

# 1. Load PDF
pdf_path = "/Users/chetansoni/Desktop/Gen Ai-25Nov/technova_employee_rulebook_12_pages.pdf"
doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text() + "\n"

print("PDF Loaded Successfully!")

# 2. Clean Data
text = re.sub(r'\s+', ' ', text)
text = text.strip()

print("\nData Cleaning Completed!")
print("Total Characters:", len(text))

# 3. Chunking
chunk_size = 500
overlap = 50

chunks = []
start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap

# 4. Display Results
print("\nTotal Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)

# 5. Save Chunks
with open("chunks.txt", "w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks):
        file.write(f"\n--- Chunk {i+1} ---\n")
        file.write(chunk + "\n")

print("\nChunks saved successfully!")


from sentence_transformers import SentenceTransformer
import numpy as np

# 6. Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 7. Generate Embedding
embeddings = model.encode(chunks)

# 8. Save Embeddings
np.save("embeddings.npy", embeddings)

print("Embeddings Generated Successfully!")
print("Total Embeddings:", len(embeddings))
print("Embedding Dimensions:", embeddings.shape)


# 9. Create FAISS Vector Database

import faiss

# Get embedding dimensions
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add embeddings to database
index.add(embeddings.astype("float32"))

# Save vector database
faiss.write_index(index, "faiss_index.bin")

print("\nFAISS Vector Database Created!")
print("Total Vectors:", index.ntotal)
print("Vector Dimensions:", dimension)


# 11. Retriever

# Load FAISS Database
index = faiss.read_index("faiss_index.bin")

# Take user question
query = input("\nAsk your question: ")

# Convert question into embedding
query_embedding = model.encode([query])

# Search top 3 relevant chunks
distances, results = index.search(
    query_embedding.astype("float32"), 3
)

# Display retrieved chunks
print("\nRelevant Results:")

for i, idx in enumerate(results[0]):
    print(f"\nResult {i+1}")
    print(chunks[idx])
    print("Distance:", distances[0][i])





# 12. Hit@K Evaluation

def evaluate_retriever(query, relevant_chunks):

    # Generate query embedding
    query_embedding = model.encode([query])

    # Search top 5 chunks
    distances, results = index.search(
        query_embedding.astype("float32"), 5
    )

    retrieved_chunks = results[0].tolist()

    print("\nQuestion:", query)
    print("Retrieved Chunk IDs:", retrieved_chunks)
    print("Distances:", distances[0].tolist())
    print("Relevant Chunk IDs:", relevant_chunks)

    # Calculate Hit@1, Hit@3 and Hit@5
    hit1 = int(any(
        idx in relevant_chunks for idx in retrieved_chunks[:1]
    ))

    hit3 = int(any(
        idx in relevant_chunks for idx in retrieved_chunks[:3]
    ))

    hit5 = int(any(
        idx in relevant_chunks for idx in retrieved_chunks[:5]
    ))

    print("Hit@1:", hit1)
    print("Hit@3:", hit3)
    print("Hit@5:", hit5)

    return hit1, hit3, hit5


# Test Dataset
# Replace example IDs with verified relevant chunk IDs

test_queries = [
    {
        "question": "What is the company's leave policy?",
        "relevant_chunks": [10]
    },
    {
        "question": "What are employee responsibilities?",
        "relevant_chunks": [15]
    },
    {
        "question": "What are the working hours?",
        "relevant_chunks": [20]
    }
]

# Run Evaluation

total_hit1 = 0
total_hit3 = 0
total_hit5 = 0

for item in test_queries:

    h1, h3, h5 = evaluate_retriever(
        item["question"],
        item["relevant_chunks"]
    )

    total_hit1 += h1
    total_hit3 += h3
    total_hit5 += h5

total = len(test_queries)

print("\n========== FINAL EVALUATION ==========")

print("Hit@1:", round(total_hit1 / total * 100, 2), "%")
print("Hit@3:", round(total_hit3 / total * 100, 2), "%")
print("Hit@5:", round(total_hit5 / total * 100, 2), "%")
