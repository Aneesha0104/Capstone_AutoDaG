from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
import os
from tools import Reader_Tool, Code_Runner, code_executor, knowledge_base,Folder_File_Getter,Insights
import litellm
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.memory import ShortTermMemory
import glob
from task_output import *
import logging
import sys

os.environ['LITELLM_LOG'] = 'DEBUG'

litellm.set_verbose = True



load_dotenv('.env',override=True)

llm = LLM(model='gemini/gemini-2.0-flash-lite',api_key=os.getenv('GEMINI_API_KEY'))
code_llm = llm 
with open('knowledge\\Agent Instructions.txt','r') as f:
    agent_instructions = f.read()
# Agents 
"""Preprocessing AGENTS"""
preprocess_agent = Agent(
            role='Data Reader',
            goal='Read and analyze data file structure',
            backstory="""Expert data reader specialized in understanding data structure 
                        and providing comprehensive initial analysis.  Generate a summary or description about the dataset.""",
            tools=[Reader_Tool(),Folder_File_Getter()],  # Custom tools the agent can use
            verbose=True,  # Enable detailed logging
            llm=llm
        )

preprocess_code_agent = Agent(
            role='Code Writer and Executor',
            goal="Generate python code and execute the generated code using the tool.",
            backstory="""As a expert Python programmer specialized in Machine Learning, data preprocessing, and pandas operations.
You are tasked to write a python code which will read a csv/excel file and then perform necessary preprocessing operations like handling null values, etc.

#Guideline for different datatypes:
- Datatype: Numeric
    -- To handle null values: Take mean
    -- To handle outlier: Perform IQR
    -- Perform binning if numeric data is continuous

- Datatype: Categorical 
    -- To handle null values: Take mode[0]""",
            verbose=True,
            llm=code_llm,
            knowledge_sources=[TextFileKnowledgeSource(file_paths=['Data preprocessing Knowledge Base.txt'],collection_name='Capstone')],
            embedder_config={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": os.getenv('GEMINI_API_KEY'),
                }
            }
        )

"""Plot suggestor - Insight Agent"""
plot_suggestor = Agent(
    role='Plot Suggestor and Insights generator',
    goal="- Read the updated dataset analyse the data and then provide plot suggestions,possible relationship between datas or columns.\n -To provide meaningful insights about the dataset, relationship between different columns, what type of plots can be generated to understand the underlying information of the data.",
    backstory="""As the greatest Data Analyst cum Scientist in the history of mankind, you are tasked with a simple problem:
                - Read the preprocessed dataset.
                - Analyze the dataset and find patterns across the columns.
                - Understand the relationships between the columns.
                - Generate insights and suggest what types of plots (e.g., scatter plots, histograms, box plots etc) can best visualize the data.
                - You must also provide the exact column name that should be used in the axis for the plot.
                """,
    verbose=True,
    llm=code_llm,
    tools=[Reader_Tool(),Folder_File_Getter()],
    knowledge_sources=[TextFileKnowledgeSource(file_paths=['Plot Suggestion Knowledge Base.txt'],collection_name='Capstone')],
    embedder_config={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv('GEMINI_API_KEY'),
        }
    }

)
"""Plot suggestor critic"""
plot_suggestor_critic = Agent(
    role='Plot Suggestor Critic',
    goal="Critique the plot suggestions provided by the Plot Suggestor Agent.",
    backstory=f"""You are an expert in data visualization and plot suggestion.
                - Critique the plot suggestions provided by the Plot Suggestor Agent.
                - Provide feedback on the plot suggestions.
                - Suggest the best possible plot for the given dataset.
                - Provide the exact column name that should be used in the axis for the plot.
                {agent_instructions}
                """,
    verbose=True,
    llm=code_llm,
    tools=[Reader_Tool(),Folder_File_Getter()],
    knowledge_sources=[TextFileKnowledgeSource(file_paths=['Agent knowledge base.txt'],collection_name='Capstone')],
    embedder_config={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv('GEMINI_API_KEY'),
        }
    }    
)

"""Plot generator/ Dashboard generator Agent"""
plot_generator = Agent(
    role='Plot Generator / Dashboard Creator',
    goal="Generate Python code that creates visualizations (charts/dashboards) based on insights and human-specified suggestions, and save all charts locally.",
    backstory="""You are an expert Python developer with a deep knowledge of data visualization using matplotlib and Plotly.
                Your job is to:
                - Read the insights provided by the Plot Suggestor Agent.
                - Generate Python code to create the required visualizations (scatter plots, histograms, box plots, or interactive dashboards).
                - Ensure the code saves all the generated charts or dashboards locally (as image files for matplotlib or HTML files for Plotly).
                - Include error handling and clear inline comments for clarity.
                """,
    verbose=True,
    llm=code_llm,
    tools=[Reader_Tool(),Folder_File_Getter()]
)


"""Generated plot Insight Agent"""
visualization_report_agent = Agent(
    role='Visualization Report Generator',
    goal="Analyze a folder containing visualizations using a specialized tool, extract detailed descriptions of each plot, and compile a structured report.",
    backstory="""You are an expert in data visualization and report generation.  
                Your primary responsibilities include:  
                - Using a dedicated tool to analyze a folder of visualizations and extract meaningful descriptions for each plot.  
                - Compiling all extracted insights into a structured, well-formatted report.  
                - Ensuring the report is clear, concise, and provides actionable insights based on the visualizations.  
                - Handling various plot types, including scatter plots, histograms, box plots, and dashboards.  
                - Implementing error handling and ensuring smooth workflow execution.  
                """,
    verbose=True,
    llm=code_llm,
    tools=[Insights()]  # Assuming this tool extracts plot descriptions
)



def get_files_in_folder(folder_path):
    """Get all CSV and Excel files in the specified folder"""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    excel_files.extend(glob.glob(os.path.join(folder_path, "*.xls")))
    return csv_files + excel_files

"""TASKS"""
reader_task = Task(
    name="Data Reader",
    description="Analyze the uploaded data files in the folder: ```{folder_path}```. Process each file **individually using the tool**. Use the Folder_File_Getter tool to get the list of files in the folder. And then use the Reader_Tool to read the csv/excel files. Finally  generate a summary or description about the dataset.",
    agent=preprocess_agent,
    expected_output="The analysis results of each data file, including null values, description, and data structure, generate a summary or description about the dataset.",
    output_pydantic=DataAnalysisResult
)

preprocessing_code_task = Task(
    name="Preprocessing code writter",
    description="""Based on the data analysis provided, generate Python code that:
    1. Reads all files from the folder: ```{folder_path}```
    2. Implements appropriate preprocessing steps (handling missing values, data types, etc.) for each file
    3. Saves the preprocessed data back to the same folder with prefix 'updated_[original_filename]'
    
    **Guidelines:**
        - If a column is only of numerical values then fill the null values using the mean of different groups using group by function.
        - Process each file individually with the necessary required preprocessing.
        - Make sure to save all outputs back to the original folder path.
        - Ensure the code is well-documented and handles errors appropriately.
        - Your code will be executed directly so don't provide any further comments without commenting them.""",
    agent=preprocess_code_agent,
    context=[reader_task],
    callback=code_executor,
    expected_output="A well structured and a fully working Python preprocessing code. Code should be enclosed with ```python ```"
)

plot_suggestor_task = Task(
    name="Plot Suggestion Task",
    description="Read the updated datasets from the folder path ```{folder_path}```,[read all the files from the folder using Folder_File_Getter and then read the updated datasets using Reader_Tool] analyze the data to identify relationships between columns, and suggest appropriate plot types (e.g., scatter, histogram, box) for visualization.",
    agent=plot_suggestor,
    # human_input=True,
    expected_output="A detailed insights of each dataset. -[S.NO] [Plotname]: [columns to be used with proper axis(x,y,z... based on the plot)]-[Why and what might be understand from this plot etc]",
    output_pydantic=PlotSuggestions,
    context=[reader_task]
)

plot_suggestor_critic_task = Task(
    name="Plot Suggestor Critic Task",
    description="Critique the plot suggestions provided by the Plot Suggestor Agent.",
    agent=plot_suggestor_critic,
    context=[plot_suggestor_task,reader_task],
    expected_output="A detailed critique of the plot suggestions provided by the Plot Suggestor Agent.Along with the proper plot suggestions.",
    output_pydantic=PlotSuggestionCritique
)

plot_generator_task = Task(
    name="Plot Generator Task",
    description="""Using the preprocessed datasets from the folder ```{folder_path}``` and insights from the Plot Suggestor Task, generate Python code that:
    1. Creates the suggested visualizations using libraries such as matplotlib or Plotly.
    2. Saves all generated charts to the same folder (e.g., image files for matplotlib plots, HTML files for Plotly dashboards).
    3. Includes appropriate error handling and documentation within the code.
    
    **Note:** If specific human input is provided for chart types or design, integrate those instructions into the visualization code.
    """,
    agent=plot_generator,
    context=[plot_suggestor_critic_task],
    callback=code_executor,
    expected_output="""Well-documented Python code that reads the updated - preprocessed dataset(s) and generates - saves the required visualizations based on the provided insights. Code should be enclosed with ```python ```"""
    )

visualization_report_task = Task(
    name="Visualization Report Task",
    description="""Analyze the generated visualizations in the folder ```{folder_path}``` and create a comprehensive report:
    1. Use the Insights tool to analyze each plot in the folder
    2. Provide detailed descriptions of what each visualization shows
    3. Identify key patterns, trends, and insights visible in the plots
    4. Compile all insights into a structured, well-formatted report
    viusalizations
    The report should be clear, concise, and provide actionable insights based on the .
    """,
    agent=visualization_report_agent,
    context=[plot_generator_task],
    expected_output="""A comprehensive report analyzing all visualizations, including detailed descriptions of each plot, key insights, and actionable recommendations."""
)

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
    agents=[preprocess_agent, preprocess_code_agent, plot_suggestor,plot_suggestor_critic, plot_generator, visualization_report_agent],
    tasks=[reader_task, preprocessing_code_task, plot_suggestor_task, plot_suggestor_critic_task, plot_generator_task, visualization_report_task],
    verbose=True
)

# The folder name is provided as input instead of specific file path
# Files will be discovered within this folder
folder_name = "dhb1f0sk-13kbsfv92"  # This can be replaced with user input
results = master_crew.kickoff(inputs={"folder_path": folder_name, "user_input": ""})
print(results)
