import gensim.downloader

glove_model = gensim.models.KeyedVectors.load("glove_model.npy")
# Function to recommend similar words
def recommend_similar_words(word, topn=5):
    try:
        similar_words = glove_model.most_similar(word, topn=15)
     
        return similar_words
    except KeyError:
        return f"'{word}' is not in the vocabulary."

# Call the function
print(recommend_similar_words('king'))