
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



