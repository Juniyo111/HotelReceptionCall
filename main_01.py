import nltk
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Function to load data from text files
def load_data(questions_file, responses_file):
    with open(questions_file, 'r', encoding='utf-8') as qf:
        questions = qf.readlines()
    with open(responses_file, 'r', encoding='utf-8') as rf:
        responses = rf.readlines()
    return [q.strip() for q in questions], [r.strip() for r in responses]

# Function to preprocess text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return filtered_tokens

# Load the questions and answers for training
questions, responses = load_data('[Dataset] Module27(ques).txt', '[Dataset] Module27 (ans).txt')

# Combine questions and responses for training
documents = questions + responses
tagged_data = [TaggedDocument(words=preprocess(doc), tags=[str(i)]) for i, doc in enumerate(documents)]

# Doc2Vec 모델 학습
model = Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.00025, min_count=1, dm=1)
model.build_vocab(tagged_data)

for epoch in range(200):
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

# 모델 저장
model.save("d2v.model")
print("Model Saved")

# Function to find the best matching response
def get_response(user_input, model, questions, responses):
    user_tokens = preprocess(user_input)
    user_vector = model.infer_vector(user_tokens).reshape(1, -1)
    
    max_sim = -1
    best_response = "Sorry, I don't understand that."
    
    for i, question in enumerate(questions):
        question_tokens = preprocess(question)
        question_vector = model.infer_vector(question_tokens).reshape(1, -1)
        sim = cosine_similarity(user_vector, question_vector)[0][0]
        
        if sim > max_sim:
            max_sim = sim
            best_response = responses[i]
            
    return best_response

# Load the Doc2Vec model
model = Doc2Vec.load("d2v.model")

print("Hotel Reception Chatbot is ready. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    response = get_response(user_input, model, questions, responses)
    print(f"Bot: {response}")
