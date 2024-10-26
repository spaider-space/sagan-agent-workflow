import json
import logging
from typing import List
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AnyMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field

'''LOCAL IMPORTS'''
from schemas import State
from prompts.prompts import *
from models.chatgroq import BuildChatGroq, BuildChatOpenAI

'''IMPORT ALL TOOLS HERE AND CREATE LIST OF TOOLS TO BE PASSED TO THE AGENT.'''
from tools.script_executor import run_script
from tools.file_tree import get_file_tree
from tools.web_tool import web_search_tool
from tools.query_chromadb import query_chromadb

load_dotenv()

terminal_tools = [run_script, get_file_tree]
research_tools = [query_chromadb]

'''LLM TO USE'''
# MODEL = "llama-3.1-70b-versatile"
# MODEL = "llama-3.1-8b-instant"
# MODEL = "gemma2-9b-it"
# MODEL = "llama3-groq-70b-8192-tool-use-preview"
# MODEL = "mixtral-8x7b-32768"
MODEL = "gpt-4o"
# llm = BuildChatGroq(model=MODEL, temperature=0)
llm = BuildChatOpenAI(model=MODEL, temperature=0)

llm_with_terminal_tools = llm.bind_tools(terminal_tools)
llm_with_research_tools = llm.bind_tools(research_tools)

class PromptParserOutput(BaseModel):
    """Ensure that this is the output of the prompt_parser node."""
    project_title: str = Field(description="The title of the project.")
    project_description: str = Field(description="The description of the project based on the project title.")

class AbstractQuestionsGeneratorOutput(BaseModel):
    """Ensure that this is the output of the abstract_questions_generator node."""
    abstract_questions: list[str] = Field(description="A list of questions that may help the Agent understand the project better.")

class AbstractAnswersGeneratorOutput(BaseModel):
    """Ensure that this is the output of the abstract_answers_generator node."""
    abstract_qa_pairs: dict[str, str] = Field(description="A dictionary of questions and answers.")
    abstract_text: str = Field(description="A summary of the project based on the answers to the questions.")

class SectionTopicExtractorOutput(BaseModel):
    """Ensure that this is the output of the section_topic_extractor node."""
    section_topics: list[str] = Field(description="A list of sections and topics that need to be filled in the template.")

class SectionWiseQuestionGeneratorOutput(BaseModel):
    """Ensure that this is the output of the section_wise_question_generator node."""
    section_questions: dict[str, list[str]] = Field(description="A dictionary of sections and their corresponding questions.")


terminal_tools_node = ToolNode(terminal_tools)
research_tools_node = ToolNode(research_tools)

def prompt_parser(state: State) -> State:
    """
    Given a user prompt, this node parses the prompt to extract the project title and description based on the project title.
    """
    print("################ PROMPT PARSER BEGIN #################")
    system_prompt = SystemMessage(PROMPT_PARSER_PROMPT)
    state["messages"].append(system_prompt)

    try:
        response = llm.invoke(state["messages"])
        print(f"Response content: {response.content}")
        print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(PromptParserOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'project_title') or not hasattr(structured_response, 'project_description'):
            raise ValueError("Project title or description not found in the structured output.")

        # print(f"Response content: {response.content}")
        print(f"Project title: {structured_response.project_title}")
        print(f"Project description: {structured_response.project_description}")
        print("################ PROMPT PARSER END #################")
        return {"messages": [response], "project_title": structured_response.project_title, "project_description": structured_response.project_description}

    except Exception as e:
        print(f"Error occurred: {e}")
        print("################ PROMPT PARSER END #################")
        return {"messages": [str(e)], "project_title": None, "project_description": None}
    
def abstract_questions_generator(state: State) -> State:
    """
    Given the project title and description, this node creates a list of questions that may help it understand the project better. The answers to these questions will then be used to create a project abstract.
    """
    print("################ ABSTRACT QUESTIONS GENERATOR BEGIN #################")
    project_title = state["project_title"]
    project_description = state["project_description"]
    system_prompt = SystemMessage(ABSTRACT_QUESTIONS_GENERATOR_PROMPT.format(project_title=project_title, project_description=project_description))

    state["messages"].append(system_prompt)

    try:
        response = llm.invoke(state["messages"])
        print(f"Response content: {response.content}")
        print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(AbstractQuestionsGeneratorOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'abstract_questions'):
            raise ValueError("Abstract questions not found in the structured output.")

        # print(f"Response content: {response.content}")
        print(f"Abstract questions: {structured_response.abstract_questions}")
        print("################ ABSTRACT QUESTIONS GENERATOR END #################")
        return {"messages": [response], "abstract_questions": structured_response.abstract_questions}

    except Exception as e:
        print(f"Error occurred: {e}")
        print("################ ABSTRACT QUESTIONS GENERATOR END #################")
        return {"messages": [str(e)], "abstract_questions": None}

def abstract_answers_generator(state: State) -> State:
    """
    Given the list of questions, this node creates answers to the questions generated by the abstract_questions_generator node.
    """
    print("################ ABSTRACT ANSWERS GENERATOR BEGIN #################")
    abstract_questions = state["abstract_questions"]
    system_prompt = SystemMessage(ABSTRACT_ANSWERS_GENERATOR_PROMPT.format(questions_list=abstract_questions))
    state["messages"].append(system_prompt)

    try:
        # Use the research tools to actually query the database
        qa_pairs = {}
        for question in abstract_questions:
            result = query_chromadb(
                "C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\ingest_data\\mychroma_db",
                "sentence-transformers/all-MiniLM-L6-v2",
                question
            )
            answer = llm.invoke(result)
            qa_pairs[question] = answer.content

        # Now use the LLM to generate an abstract based on the retrieved answers
        abstract_prompt = f"Based on the following question-answer pairs, generate a concise abstract for the JEDI Cloud project:\n\n"
        for q, a in qa_pairs.items():
            abstract_prompt += f"Q: {q}\nA: {a}\n\n"
        
        abstract_response = llm.invoke(abstract_prompt)
        abstract_text = abstract_response.content

        structured_response = AbstractAnswersGeneratorOutput(
            abstract_qa_pairs=qa_pairs,
            abstract_text=abstract_text
        )

        print(f"Abstract QA pairs: {structured_response.abstract_qa_pairs}")
        print(f"Abstract text: {structured_response.abstract_text}")
        print("################ ABSTRACT ANSWERS GENERATOR END #################")
        return {"messages": [abstract_response], "abstract_text": structured_response.abstract_text}

    except Exception as e:
        print(f"Error occurred: {e}")
        print("################ ABSTRACT ANSWERS GENERATOR END #################")
        return {"messages": [str(e)], "abstract_text": None}

def section_topic_extractor(state: State) -> State:
    """
    This node extracts the topics for each section of the project from the template pdf given by the user.
    """
    print("################ SECTION TOPIC EXTRACTOR BEGIN #################")
    system_prompt = SystemMessage(SECTION_TOPIC_EXTRACTOR_PROMPT)
    state["messages"].append(system_prompt)

    try:
        response = llm_with_research_tools.invoke(state["messages"])
        print(f"Response content: {response.content}")
        print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(SectionTopicExtractorOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'section_topics'):
            raise ValueError("Section topics not found in the structured output.")

        # print(f"Response content: {response.content}")
        print(f"Section topics: {structured_response.section_topics}")
        print("################ SECTION TOPIC EXTRACTOR END #################")
        return {"messages": [response], "section_topics": structured_response.section_topics}

    except Exception as e:
        print(f"Error occurred: {e}")
        print("################ SECTION TOPIC EXTRACTOR END #################")
        return {"messages": [str(e)], "section_topics": None}

def section_wise_question_generator(state: State) -> State:
    """
    Given the list of sections, this node creates a list of questions for each section.
    """
    print("################ SECTION WISE QUESTION GENERATOR BEGIN #################")
    section_topics = state["section_topics"]
    system_prompt = SystemMessage(SECTION_WISE_QUESTION_GENERATOR_PROMPT.format(section_topics=section_topics))

    state["messages"].append(system_prompt)

    try:
        response = llm.invoke(state["messages"])
        print(f"Response content: {response.content}")
        print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(SectionWiseQuestionGeneratorOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'section_questions'):
            raise ValueError("Section questions not found in the structured output.")

        # print(f"Response content: {response.content}")
        print(f"Section questions: {structured_response.section_questions}")
        print("################ SECTION WISE QUESTION GENERATOR END #################")
        return {"messages": [response], "section_questions": structured_response.section_questions}

    except Exception as e:
        print(f"Error occurred: {e}")
        print("################ SECTION WISE QUESTION GENERATOR END #################")
        return {"messages": [str(e)], "section_questions": None}

def abstract_answers_generator(state: State) -> State:
    """
    Given the list of questions, this node creates answers to the questions generated by the abstract_questions_generator node.
    """
    print("################ ABSTRACT ANSWERS GENERATOR BEGIN #################")
    abstract_questions = state["abstract_questions"]
    system_prompt = SystemMessage(ABSTRACT_ANSWERS_GENERATOR_PROMPT.format(questions_list=abstract_questions))
    state["messages"].append(system_prompt)

    try:
        # Use the research tools to actually query the database
        qa_pairs = {}
        for question in abstract_questions:
            result = query_chromadb(
                "C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\ingest_data\\mychroma_db",
                "sentence-transformers/all-MiniLM-L6-v2",
                question
            )
            answer = llm.invoke(result)
            qa_pairs[question] = answer.content

        # Now use the LLM to generate an abstract based on the retrieved answers
        abstract_prompt = f"Based on the following question-answer pairs, generate a concise abstract for the JEDI Cloud project:\n\n"
        for q, a in qa_pairs.items():
            abstract_prompt += f"Q: {q}\nA: {a}\n\n"
        
        abstract_response = llm.invoke(abstract_prompt)
        abstract_text = abstract_response.content

        structured_response = AbstractAnswersGeneratorOutput(
            abstract_qa_pairs=qa_pairs,
            abstract_text=abstract_text
        )

        print(f"Abstract QA pairs: {structured_response.abstract_qa_pairs}")
        print(f"Abstract text: {structured_response.abstract_text}")
        print("################ ABSTRACT ANSWERS GENERATOR END #################")
        return {"messages": [abstract_response], "abstract_text": structured_response.abstract_text}

    except Exception as e:
        print(f"Error occurred: {e}")
        print("################ ABSTRACT ANSWERS GENERATOR END #################")
        return {"messages": [str(e)], "abstract_text": None}

def reporter(state: State):
    """
    This function is used to talk to the user like a regular chatbot.
    """
    return {"messages": llm.invoke(state["messages"])}




