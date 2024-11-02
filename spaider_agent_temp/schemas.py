from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv()

class State(MessagesState):
    project_title: str
    project_description: str
    abstract_questions: list[str]
    abstract_text: str
    section_topics: list[str]
    section_questions: dict[str, list[str]] # key = section_topic, value = list of questions.
    section_answers: dict[str, list[str]] # key = section_topic, value = list of answers.
    plan: dict[str, list[str]] # key = section_topic, value = list of steps.
    draft: str
