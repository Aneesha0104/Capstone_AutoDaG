import streamlit as st
import os
import glob
from pathlib import Path
from io import BytesIO
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from tools import Reader_Tool, Code_Runner, code_executor, knowledge_base, Folder_File_Getter, Insights
import litellm
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.memory import ShortTermMemory
from task_output import *
import logging
import sys

# Set up logging for debugging
os.environ['LITELLM_LOG'] = 'DEBUG'
litellm.set_verbose = True

# Load environment variables from .env file
load_dotenv('.env', override=True)

# Initialize LLM
llm = LLM(model='gemini/gemini-2.0-flash-lite', api_key=os.getenv('GEMINI_API_KEY'))
code_llm = llm 

# Load agent instructions
with open('knowledge\\Agent Instructions.txt', 'r') as f:
    agent_instructions = f.read()

# Agents
preprocess_agent = Agent(
    role='Data Reader',
    goal='Read and analyze data file structure',
    backstory="""Expert data reader specialized in understanding data structure 
                and providing comprehensive initial analysis.  Generate a summary or description about the dataset.""",
    tools=[Reader_Tool(), Folder_File_Getter()],
    verbose=True,
    llm=llm
)

preprocess_code_agent = Agent(
    role='Code Writer and Executor',
    goal="Generate python code and execute the generated code using the tool.",
    backstory="""As an expert Python programmer specialized in Machine Learning, data preprocessing, and pandas operations.
    You are tasked to write python code which will read a csv/excel file and then perform necessary preprocessing operations like handling null values, etc.""",
    verbose=True,
    llm=code_llm,
    knowledge_sources=[TextFileKnowledgeSource(file_paths=['Data preprocessing Knowledge Base.txt'], collection_name='Capstone')],
    embedder_config={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv('GEMINI_API_KEY'),
        }
    }
)

# Define tasks for the agents (unchanged from your original code)
reader_task = Task(
    name="Data Reader",
    description="Analyze the uploaded data files in the folder: ```{folder_path}```.",
    agent=preprocess_agent,
    expected_output="The analysis results of each data file.",
    output_pydantic=DataAnalysisResult
)

preprocessing_code_task = Task(
    name="Preprocessing code writter",
    description="""Generate Python code for preprocessing uploaded files.""",
    agent=preprocess_code_agent,
    context=[reader_task],
    callback=code_executor,
    expected_output="A well-structured and fully working Python preprocessing code."
)

# Define your master crew (crew of agents) - as before
master_crew = Crew(
    memory=True,
    embedder={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv('GEMINI_API_KEY'),
        }
    },
    llm=llm,
    agents=[preprocess_agent, preprocess_code_agent],
    tasks=[reader_task, preprocessing_code_task],
    verbose=True
)

# The folder name is provided as input instead of specific file path
# Files will be discovered within this folder
folder_name = "1kjbjkb923-13813kbds"  # This can be replaced with user input
results = master_crew.kickoff(inputs={"folder_path": folder_name, "user_input": ""})
print(results)
