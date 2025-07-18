import json
from datetime import datetime

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
def clean_isoformat(dt_str):
    # Remove spaces around `:` to fix bad ISO strings
    return dt_str.replace(" ", "")

def find_question_label(question_id, questions):
    for question in questions:
        if question['id'] == question_id:
            return question['label'], question['questionOptions']
    return None, []

def find_option_label(option_id, options):
    for option in options:
        if option['id'] == option_id:
            return option['label']
    return "Unknown"

def process_assessment(report):
    attempted_at = report.get("attemptedAt", "Unknown Date")
    try:
        clean_attempted = clean_isoformat(attempted_at).replace("Z", "")
        attempted_date = datetime.fromisoformat(clean_attempted).strftime("%B %d, %Y %H:%M:%S")
    except ValueError:
        attempted_date = attempted_at  # fallback to raw
        
    assessment = report.get("assessment", {})
    description = assessment.get("description", "")
    insight = report.get("insight", "").strip()

    questions = assessment.get("questions", [])
    answers = report.get("answers", [])

    markdown = []
    markdown.append(f"# Assessment Attempt\n\n**Date:** {attempted_date}\n")
    markdown.append(f"## Description\n{description}\n")
    markdown.append(f"## Insight\n{insight}\n")
    markdown.append(f"## Answers")

    for answer in answers:
        question_id = answer.get("questionId")
        option_id = answer.get("questionOptionId")

        question_label, question_options = find_question_label(question_id, questions)
        answer_label = find_option_label(option_id, question_options)

        if question_label and answer_label:
            markdown.append(f"\n**Q:** {question_label}\n**A:** {answer_label}")

    markdown.append("\n---\n")
    return '\n'.join(markdown)

def process_assessments_to_markdown(assessments_data):
    """
    Convert assessments JSON data to a list of markdown strings.
    Each assessment becomes a separate markdown string.
    
    Args:
        assessments_data: List of assessment reports from JSON
        
    Returns:
        List[str]: List of markdown strings, one per assessment
    """
    markdown_assessments = []
    for report in assessments_data:
        md = process_assessment(report)
        markdown_assessments.append(md)
    return markdown_assessments
