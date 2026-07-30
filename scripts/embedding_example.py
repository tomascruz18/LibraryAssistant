from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load model (only once)
model = SentenceTransformer("BAAI/bge-m3")

paper1 = """
Optimization of regenerative cooling channels
using CFD and surrogate models.
"""

paper2 = """
CFD optimization of cooling passages
for liquid rocket engines.
"""

paper3 = """
Shakespeare's influence on modern literature.
"""
embeddings = model.encode([paper1, paper2, paper3])

similarity = cosine_similarity(embeddings)

print(similarity)

text1 = """
1. Method of Characteristics paper
"""

text2 = """
2. Other Method of Characteristics paper
"""

text3 = """
3. Bayesian calibration paper
"""

search = "method of characteristics"

search2 = "Represent this sentence for searching relevant passages: method of characteristics"

embeddings = model.encode([text1, text2, text3, search, search2])

similarity = cosine_similarity(embeddings)

print(similarity)