# Sample input text
context = """
Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality. The distinction between the former and the latter categories is often revealed by the acronym chosen. 'Strong' AI is usually labeled as AGI (Artificial General Intelligence) while attempts to emulate 'natural' intelligence have been called ABI (Artificial Biological Intelligence).
"""

# Test the post_mca_questions function
mca_questions = post_mca_questions(context, num_questions=3)

# Display generated multiple-choice questions
for i, question in enumerate(mca_questions, 1):
    print(f"Question {i}:")
    print(question)
