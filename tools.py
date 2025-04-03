from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import io
import subprocess
import time
import sys
import logging
import glob
import asyncio
import concurrent.futures

logging.basicConfig(level=logging.INFO,filename='logs_logger.txt')

def knowledge_base():
    files = glob.glob('knowledge_base\\*')
    return files

class Reader_Tool(BaseTool):
    def Input_param():
        class MyToolInput(BaseModel):
            """Input scheme - file_path"""
            file_path: str = Field(...,description='single file_path of the excel/csv file')
        return MyToolInput

    name:str = "DatasetReader"
    description:str = "Reads an excel/csv file [can process one file at one time, if multiple files then call individually] from the given file_path and returns information about the dataset -> null-values, description, first 5-10 rows, info, column types etc. "
    args_schema: Type[BaseModel] = Input_param()
    def _run(self, file_path:str) -> str :
        def get_describe(file_path:str,dataframe:pd.DataFrame):
            return f"Description of the Dataframe - {file_path} :- ```{dataframe.describe()}```\n"
        
        def get_info(file_path: str, dataframe: pd.DataFrame):
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            info_str = buffer.getvalue()
            buffer.close()
            return f"Information of the Dataframe - {file_path} :- ```{info_str}```\n"

        def get_null(file_path: str, dataframe: pd.DataFrame):
            total_missing = dataframe.isnull().sum().sum()
            missing_by_column = dataframe.isnull().sum().to_dict()
            missing_percentage = (dataframe.isnull().sum() / len(dataframe) * 100).to_dict()

            return (
                f"Null values of the Dataframe - ```{file_path} :-\n"
                f"Total Missing: {total_missing}\n"
                f"Missing by Column: {missing_by_column}\n"
                f"Missing Percentage: {missing_percentage}```\n"
            )

        def get_head(file_path:str,dataframe:pd.DataFrame):
            return f"Head of the Dataframe - {file_path} :- ```{dataframe.head(5)}```\n" 
        
        # file_path = file_path[0]
        print(file_path)
        try:
                
            try:    
                df = pd.read_excel(file_path)
                return f"""After reading the file {file_path}:\n \
                    {get_describe(file_path=file_path,dataframe=df)} \
                    {get_info(file_path=file_path,dataframe=df)} \
                    {get_null(file_path=file_path,dataframe=df)}\
                    {get_head(file_path=file_path,dataframe=df)}
                    """
            except Exception as e:
                df = pd.read_csv(file_path)
                return f"""After reading the file {file_path}:\n \
                    {get_describe(file_path=file_path,dataframe=df)} \
                    {get_info(file_path=file_path,dataframe=df)} \
                    {get_null(file_path=file_path,dataframe=df)}\
                    {get_head(file_path=file_path,dataframe=df)}
                    """
        except Exception as e:
            return f"""Error: Could not read the dataframe {str(e)}"""
    
    
class Code_Runner(BaseTool):
    def Input_param():
        class MyToolInput(BaseModel):
            """Input scheme - code"""
            code: str = Field(...,description='Provide the code that needs to be executed as a string')
            # code_description: str = Field(...,description='Description of the task for which code that needs to be executed [max 2 words : format separated by "_" underscores ]')
        return MyToolInput

    name:str = "Code_Executor"
    description:str = "Accepts the code as a string in the key 'code', executes the code and provides the output. If encountered an error - returns error."
    args_schema: Type[BaseModel] = Input_param()
    def _run(self, code:str) -> str :
        try:
            file_name = 'preprocessing.py'
            print(file_name)
            print(code)
            with open(file_name, "wb") as script_file:
                    script_file.write(str(code).encode('utf-8'))
            result = subprocess.run(
                [sys.executable, file_name],
                capture_output=True,
                text=True,
                check=True,timeout=30  # Raises an exception if the script returns a non-zero exit code
            )
            logging.info("Script Output:\n%s", result.stdout)
            return f"Execution Result {result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error while executing the script {e.stderr}"
        except Exception as e:
            return f"Error : {str(e)}"
        
def code_executor(code):
        def beautify_code(file_path):
            """Formats the Python file using black."""
            try:
                subprocess.run(["black", file_path], check=True)
                print(f"✅ Code formatted successfully: {file_path}")
            # except subprocess.CalledProcessError:
            #     print(f"❌ Failed to format {file_path}. Ensure 'black' is installed.")
            except Exception as e:
                print(f"❌ Error: {e}")

        def lint_code(file_path):
            """Runs pylint to check for errors in the Python file."""
            try:
                lint_result = subprocess.run(["pylint", file_path], capture_output=True, text=True)
                print(f"🔍 Pylint Output:\n{lint_result.stdout}")
            except Exception as e:
                print(f"❌ Error running pylint: {e}")
        
        try:
            code = str(code)
            file_name = 'preprocessing.py'
            code_file = code.split('```python')[1].split('```')[0]
            # print(code_file)
            code_block = code_file.encode('utf-8-sig').decode('utf-8-sig')

            code_lines = code_block.splitlines()
            cleaned_code = '\n'.join(line.rstrip() for line in code_lines)

            with open(file_name, "w", encoding='utf-8') as script_file:
                script_file.write(cleaned_code)
        
            print("\n🔹 Beautifying the code...\n")
            beautify_code(file_name)

            print("\n🔹 Running code analysis...\n")
            lint_code(file_name)
            
            result = subprocess.run(
                [sys.executable, file_name],
                capture_output=True,
                text=True,
                check=True  # Raises an exception if the script returns a non-zero exit code
            )
            time.sleep(10)
            print(f"Execution Result {result.stdout}")
        except subprocess.CalledProcessError as e:
                print("Subprocess error:", e.stderr)
                return f"Error while executing the script: {e.stderr}"
        except Exception as e:
            print(e)
            return f"Error : {str(e)}"
        
        
class Folder_File_Getter(BaseTool):
    def Input_param():
        class MyToolInput(BaseModel):
            """Input scheme - file_path"""
            folder_name: str = Field(...,description='folder name alone')
        return MyToolInput

    name:str = "Folder - file_name getter"
    description:str = "Gets a folder name as an input and outputs all the files present in the respective folder."
    args_schema: Type[BaseModel] = Input_param()
    def _run(self, folder_name:str) -> str :
        print(folder_name)
        try:
            files = glob.glob(f'{folder_name}\\*')
            return f"""Available files in the folder {folder_name} :- {files}"""
        except Exception as e:
            return f"""Error: Could not find the folder {str(e)}"""
   
class Insights(BaseTool):
    def Input_param():
        class MyToolInput(BaseModel):
            """Input scheme - file_path"""
            folder_name: str = Field(...,description='folder name alone')
            file_path: str = Field(...,description='file_path of the updated excel/csv file with the folder name aswell')
        return MyToolInput

    name:str = "Plot Insights Generator"
    description:str = "Gets a folder name and dataset file path as input, analyzes all plots in the folder, and generates detailed descriptions using Gemini API."
    args_schema: Type[BaseModel] = Input_param()
    def _run(self, folder_name:str, file_path:str) -> str :
        print(folder_name)
        try:
            # Get all image files in the folder
            image_files = []
            for ext in [".png", ".jpg", ".jpeg"]:
                image_files += glob.glob(f'{folder_name}\\*{ext}')
            
            if not image_files:
                return f"No image files found in folder: {folder_name}"
            
            # Read the dataset for context
            try:
                df = pd.read_excel(file_path)
            except Exception:
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    return f"Error reading dataset file {file_path}: {str(e)}"
            
            # Get dataset summary for context
            dataset_summary = {
                "columns": list(df.columns),
                "shape": df.shape,
                "sample_data": df.head(3).to_dict(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Import required libraries
            try:
                import google.generativeai as genai
                from PIL import Image
                import base64
                import io
                import os
                import asyncio
                import aiohttp
            except ImportError:
                return "Error: Required libraries not installed. Please install google-generativeai, Pillow, asyncio, and aiohttp."
            
            # Configure Gemini API
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                model = genai.GenerativeModel('gemini-2.0-flash')
            except Exception as e:
                return f"Error configuring Gemini API: {str(e)}"
            
            # Async function to process a single image
            async def process_image(img_path, semaphore):
                async with semaphore:  # Limit concurrent API calls
                    try:
                        # Load image
                        image = Image.open(img_path)
                        
                        # Convert image to base64 for API
                        buffered = io.BytesIO()
                        image.save(buffered, format=image.format)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Create prompt for Gemini
                        prompt = f"""
                        Analyze this data visualization image and provide a detailed description.
                        
                        Dataset context:
                        - Columns: {dataset_summary['columns']}
                        - Data sample: {str(dataset_summary['sample_data'])[:500]}
                        
                        For your analysis, include:
                        1. Type of visualization (e.g., scatter plot, bar chart, histogram)
                        2. What variables/columns are being visualized
                        3. Key patterns, trends, or insights visible in the plot
                        4. Any notable outliers or interesting data points
                        5. How this visualization relates to the dataset context
                        
                        Provide a concise but comprehensive description.
                        """
                        
                        # Call Gemini API (run in executor to avoid blocking)
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None, 
                            lambda: model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_str}])
                        )
                        
                        # Return description
                        plot_name = os.path.basename(img_path)
                        return f"## {plot_name}\n{response.text}\n"
                        
                    except Exception as e:
                        plot_name = os.path.basename(img_path)
                        return f"## {plot_name}\nError analyzing image: {str(e)}\n"
            
            # Run all image processing tasks concurrently with asyncio
            async def process_all_images():
                # Create a semaphore to limit concurrent API calls (avoid rate limiting)
                semaphore = asyncio.Semaphore(5)  # Max 5 concurrent API calls
                
                # Create tasks for all images
                tasks = [process_image(img_path, semaphore) for img_path in image_files]
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                return results
            
            # Run the async event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            plot_descriptions = loop.run_until_complete(process_all_images())
            loop.close()
            
            # Combine all descriptions
            all_descriptions = "\n".join(plot_descriptions)
            with open(f"{folder_name}\\plot_analysis_report.txt", "w") as file:
                file.write(f"""# Plot Analysis Report for {folder_name}\n\n{all_descriptions}""")
            return f"""# Plot Analysis Report for {folder_name}\n\n{all_descriptions}"""
            
        except Exception as e:
            return f"""Error analyzing plots: {str(e)}"""
   