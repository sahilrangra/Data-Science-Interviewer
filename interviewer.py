import random
from transformers import pipeline

evaluator = pipeline("text2text-generation", model="google/flan-t5-small")

questions = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is the bias-variance tradeoff?",
    "How does a decision tree work?",
    "What is cross-validation, and why is it important?",
    "Explain precision, recall, and F1-score.",
    "What are the assumptions of linear regression?",
    "What is feature engineering? Provide an example.",
    "How does gradient descent work?",
    "Explain the importance of regularization in machine learning.",
    "What are the advantages of using ensemble learning?"
]

def ask_question():
    """Randomly selects a question and asks the user."""
    question = random.choice(questions)
    print("\nInterviewer: " + question)
    return question

def evaluate_answer(user_answer, question):
    """Uses AI to evaluate the user's answer and provide feedback."""
    prompt = f"Question: {question}\nCandidate's Answer: {user_answer}\nEvaluate the answer and give a score (0-10) along with feedback."
    
    try:
        response = evaluator(prompt, max_length=200)[0]['generated_text']
        return response
    except Exception as e:
        return f"An error occurred while evaluating your answer: {e}"

def main():
    print("Welcome to the AI Data Science Interviewer!")
    while True:
        question = ask_question()
        user_answer = input("\nYour Answer: ").strip()

        if user_answer.lower() in ['exit', 'quit']:
            print("Exiting Interview. Good luck with your preparation!")
            break

        if not user_answer:
            print("Please provide an answer.")
            continue

        print("\nEvaluating your answer...\n")
        feedback = evaluate_answer(user_answer, question)
        print("AI Feedback:\n", feedback)

if __name__ == "__main__":
    main()