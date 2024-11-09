import json
import logging
from typing import List
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from colorama import init, Fore, Back, Style

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
init()

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
    section_questions: dict[str, list[str]] = Field(description="A dictionary of sections and their corresponding list of questions.")


terminal_tools_node = ToolNode(terminal_tools)
research_tools_node = ToolNode(research_tools)

def prompt_parser(state: State) -> State:
    """
    Given a user prompt, this node parses the prompt to extract the project title and description based on the project title.
    """
    print(f"{Fore.YELLOW}################ PROMPT PARSER BEGIN #################")
    system_prompt = SystemMessage(PROMPT_PARSER_PROMPT)
    state["messages"].append(system_prompt)

    try:
        response = llm.invoke(state["messages"])
        # print(f"Response content: {response.content}")
        # print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(PromptParserOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'project_title') or not hasattr(structured_response, 'project_description'):
            raise ValueError("Project title or description not found in the structured output.")

        # print(f"Response content: {response.content}")
        # print(f"Project title: {structured_response.project_title}")
        # print(f"Project description: {structured_response.project_description}")
        
        # Updating state before end-of-node logging
        state["messages"].append(response)
        state["project_title"] = structured_response.project_title
        state["project_description"] = structured_response.project_description

        print(f"\n\n\n\nstate at the end of prompt_parser: \n")
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ PROMPT PARSER END #################{Style.RESET_ALL}")
        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ PROMPT PARSER END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["project_title"] = None
        state["project_description"] = None
        return state
    
def abstract_questions_generator(state: State) -> State:
    """
    Given the project title and description, this node creates a list of questions that may help it understand the project better. The answers to these questions will then be used to create a project abstract.
    """
    print(f"{Fore.RED}################ ABSTRACT QUESTIONS GENERATOR BEGIN #################")
    project_title = state["project_title"]
    project_description = state["project_description"]
    system_prompt = SystemMessage(ABSTRACT_QUESTIONS_GENERATOR_PROMPT.format(project_title=project_title, project_description=project_description))
    state["messages"].append(system_prompt)

    try:
        response = llm.invoke(state["messages"])
        # print(f"Response content: {response.content}")
        # print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(AbstractQuestionsGeneratorOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'abstract_questions'):
            raise ValueError("Abstract questions not found in the structured output.")

        # print(f"Response content: {response.content}")
        # print(f"Abstract questions: {structured_response.abstract_questions}")

        # Updating state before end-of-node logging

        state["messages"].append(response)
        state["abstract_questions"] = structured_response.abstract_questions
        print(f"\n\n\n\nstate at the end of abstract_questions_generator: \n")
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ ABSTRACT QUESTIONS GENERATOR END #################{Style.RESET_ALL}")
        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ ABSTRACT QUESTIONS GENERATOR END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["abstract_questions"] = None
        return state

def abstract_answers_generator(state: State) -> State:
    """
    Given the list of questions, this node creates answers to the questions generated by the abstract_questions_generator node.
    """
    print(f"{Fore.BLUE}################ ABSTRACT ANSWERS GENERATOR BEGIN #################")
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
            answer = llm.invoke((f"Frame the following texts into one cohesive answer: {result}"))
            qa_pairs[question] = answer.content
            # print(f"Question: {question}\nAnswer: {answer.content} \n\n\n\n")

        # Now use the LLM to generate an abstract based on the retrieved answers
        project_title = state["project_title"]
        abstract_prompt = f"Based on the following question-answer pairs, generate a concise abstract for the {project_title} project:\n\n"
        for q, a in qa_pairs.items():
            abstract_prompt += f"Q: {q}\nA: {a}\n\n"
        
        abstract_response = llm.invoke(abstract_prompt)
        abstract_text = abstract_response.content

        structured_response = AbstractAnswersGeneratorOutput(
            abstract_qa_pairs=qa_pairs,
            abstract_text=abstract_text
        )

        # print(f"Abstract QA pairs: {structured_response.abstract_qa_pairs}")
        # print(f"\n\n\n\nAbstract text: {structured_response.abstract_text}")
        
        # Updating state before end-of-node logging
        state["messages"].append(abstract_response)
        state["abstract_text"] = structured_response.abstract_text
        print(f"\n\n\n\nstate at the end of abstract_answers_generator: \n")
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ ABSTRACT ANSWERS GENERATOR END #################{Style.RESET_ALL}")


        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ ABSTRACT ANSWERS GENERATOR END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["abstract_text"] = None
        return state

def section_topic_extractor(state: State) -> State:
    """
    This node extracts the topics for each section of the project from the template pdf given by the user.
    """
    print(f"{Fore.CYAN}################ SECTION TOPIC EXTRACTOR BEGIN #################")
    system_prompt = SystemMessage(SECTION_TOPIC_EXTRACTOR_PROMPT)
    state["messages"].append(system_prompt)

    try:
        response = llm_with_research_tools.invoke(state["messages"])
        # print(f"Response content: {response.content}")
        # print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        llm_with_structured_output = llm.with_structured_output(SectionTopicExtractorOutput)
        structured_response = llm_with_structured_output.invoke(response.content)

        if not hasattr(structured_response, 'section_topics'):
            raise ValueError("Section topics not found in the structured output.")

        # print(f"Response content: {response.content}")
        # print(f"Section topics: {structured_response.section_topics}")
        
        # Updating state before end-of-node logging
        state["messages"].append(response)
        state["section_topics"] = structured_response.section_topics
        print(f"\n\n\n\nstate at the end of section_topic_extractor: \n")
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ SECTION TOPIC EXTRACTOR END #################{Style.RESET_ALL}")


        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ SECTION TOPIC EXTRACTOR END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["section_topics"] = None
        return state

def section_wise_question_generator(state: State) -> State:
    """
    Given the list of sections, this node creates a list of questions for each section.
    """
    print(f"{Fore.MAGENTA}################ SECTION WISE QUESTION GENERATOR BEGIN #################")
    section_topics = state["section_topics"]
    system_prompt = SystemMessage(SECTION_WISE_QUESTION_GENERATOR_PROMPT.format(section_topics=section_topics))
    state["messages"].append(system_prompt)

    try:
        response = llm.invoke(state["messages"])
        
        # Parse JSON response directly into section_questions dictionary
        section_questions = json.loads(response.content)

        # Updating state before end-of-node logging
        state["messages"].append(response)
        state["section_questions"] = section_questions
        
        print(f"\n\n\n\nstate at the end of section_wise_question_generator: \n")
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ SECTION WISE QUESTION GENERATOR END #################{Style.RESET_ALL}")


        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ SECTION WISE QUESTION GENERATOR END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["section_questions"] = None
        return state

def section_wise_answers_generator(state: State) -> State:
    """
    Given the dictionary of section-wise questions, this node finds answers to the questions
    generated by the section_wise_question_generator node.
    """
    print(f"{Fore.GREEN}################ SECTION WISE ANSWERS GENERATOR BEGIN #################")
    section_questions = state.get("section_questions")
    if not section_questions:
        error_msg = "No section questions found in state. Previous node may have failed."
        print(f"Error: {error_msg}")
        print(f"################ SECTION WISE ANSWERS GENERATOR END #################{Style.RESET_ALL}")
        return {
            "messages": [SystemMessage(content=error_msg)],
            "section_answers": None
        }

    try:
        section_answers = {}
        
        for section, questions in section_questions.items():
            section_answers[section] = []
            for question in questions:
                result = query_chromadb(
                    "C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\ingest_data\\mychroma_db",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    question
                )
                answer = llm.invoke((f"Frame the following texts into one cohesive answer: {result}"))
                section_answers[section].append(answer.content)

        # Updating state before end-of-node logging
        state["messages"].append(SystemMessage(content="Section-wise answers generated successfully"))
        state["section_answers"] = section_answers

        print(f"\n\n\n\nstate at the end of section_wise_answers_generator: \n")
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ SECTION WISE ANSWERS GENERATOR END #################{Style.RESET_ALL}")
        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ SECTION WISE ANSWERS GENERATOR END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["section_answers"] = None
        return state

def plan_node(state: State):
    print(f"{Fore.LIGHTYELLOW_EX}################ PLAN NODE BEGIN #################")
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=f"Project abstract: {state['abstract_text']}\n\nSection-wise texts: {state['section_answers']}")
    ]
    response = llm.invoke(messages)

    # Updating state before end-of-node logging
    state["messages"].append(response)
    state["plan"] = response.content
    print(f"\n\n\n\nstate at the end of plan_node: \n")
    print("Messages: ")
    messages = state["messages"]
    if len(messages) >= 3:
        for message in messages[-3:]:
            print(f"{message.type}: {message.content}")
    else:
        for message in messages:
            print(f"{message.type}: {message.content}")
    for field_name, field_value in state.items():
        if field_name != "messages":
            print(f"- {field_name}: {field_value}")
    print(f"################ PLAN NODE END #################{Style.RESET_ALL}")


    return state

def generation_node(state: State):
    print(f"{Fore.LIGHTGREEN_EX}################ GENERATION NODE BEGIN #################")
    section_texts = state["section_answers"]
    plan = state["plan"]
    user_message = HumanMessage(
        content=f"\n\nHere is my plan:\n\n{plan}\n\nHere are the section-wise texts:\n\n{section_texts}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT
        ),
        user_message
        ]
    response = llm.invoke(messages)
    # print(f"\n\n\n\nDraft: {response.content}")

    # Updating state before end-of-node logging
    state["messages"].append(response)
    state["draft"] = response.content
    print(f"\n\n\n\nstate at the end of generation_node: \n")
    print("Messages: ")
    messages = state["messages"]
    output_path = "C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\output\\output.md"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(state["draft"])
    if len(messages) >= 3:
        for message in messages[-3:]:
            print(f"{message.type}: {message.content}")
    else:
        for message in messages:
            print(f"{message.type}: {message.content}")
    for field_name, field_value in state.items():
        if field_name != "messages":
            print(f"- {field_name}: {field_value}")
    print(f"################ GENERATION NODE END #################{Style.RESET_ALL}")
    return state



# def reporter(state: State):
#     """
#     This function is used to talk to the user like a regular chatbot.
#     """
#     print(f"{Fore.WHITE}################ REPORTER BEGIN #################{Style.RESET_ALL}")
#     print(f"################ REPORTER END #################{Style.RESET_ALL}")
#     return {"messages": llm.invoke(state["messages"])}




