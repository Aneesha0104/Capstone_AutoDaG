import streamlit as st
import os
import glob
from pathlib import Path
from io import BytesIO
import zipfile
from tempfile import TemporaryDirectory
from crewai import Agent, Task, Crew, LLM, Process
from task_output import DataAnalysisResult, PlotSuggestions, PlotSuggestionCritique
from main import master_crew

# Function to save uploaded files in a temporary directory
def save_uploaded_files(uploaded_files):
    temp_dir = TemporaryDirectory()
    for uploaded_file in uploaded_files:
        file_path = Path(temp_dir.name) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_dir.name

# Streamlit UI elements
st.title('Data Analysis, Preprocessing, and Plot Generation')

st.subheader("Upload Your Dataset (CSV/Excel)")
uploaded_files = st.file_uploader("Choose CSV or Excel files", accept_multiple_files=True, type=["csv", "xlsx", "xls"])

if uploaded_files:
    folder_path = save_uploaded_files(uploaded_files)
    st.write("Files uploaded successfully!")

    # Only trigger processing when the button is clicked
    if st.button('Start Processing'):
        with st.spinner('Processing your files...'):
            # Call the master crew kickoff function (process the data) only when the button is pressed
            results = master_crew.kickoff(inputs={"folder_path": folder_path, "user_input": ""})

            # Show the output results
            st.write("Process Completed!")

            # Display Data Analysis Results
            if 'data_analysis' in results:
                st.subheader("Data Analysis Results")
                st.write(results['data_analysis'])

            # Display Plot Suggestions
            if 'plot_suggestions' in results:
                st.subheader("Plot Suggestions")
                for idx, suggestion in enumerate(results['plot_suggestions']):
                    st.write(f"**{idx+1}. Plot Name**: {suggestion['plot_name']}")
                    st.write(f"**Columns**: {suggestion['columns']}")
                    st.write(f"**Reasoning**: {suggestion['reasoning']}")

            # Display Critiques of Plot Suggestions
            if 'plot_suggestions_critic' in results:
                st.subheader("Plot Suggestion Critiques")
                for critique in results['plot_suggestions_critic']:
                    st.write(f"**Critique**: {critique['critique']}")
                    st.write(f"**Suggested Plot**: {critique['suggested_plot']}")

            # Display Visualizations
            if 'visualizations' in results:
                st.subheader("Generated Visualizations")
                for viz in results['visualizations']:
                    # Assuming images or HTML files are stored and accessible
                    img_path = os.path.join(folder_path, viz['image_file'])
                    if img_path.endswith(".html"):
                        st.components.v1.html(open(img_path).read(), height=600)
                    else:
                        st.image(img_path, caption=viz['caption'])

            # Display the visualization report
            if 'visualization_report' in results:
                st.subheader("Visualization Report")
                st.write(results['visualization_report'])

else:
    st.warning("Please upload at least one CSV or Excel file.")
