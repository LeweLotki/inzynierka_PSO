from subprocess import (
    run, CalledProcessError
)

from os import path

def run_analysis(notebook_file_name: str):
    
    notebook_folder_path = './data_analysis/notebooks/'
    notebook_file_path = notebook_folder_path + notebook_file_name
    
    try:
        if not path.exists(notebook_file_path):
            print(f"Error: Notebook file '{notebook_file_path}' not found.")
            return
        # Use the subprocess module to run the jupyter notebook command
        run(['poetry', 'run', 'jupyter', 'notebook', notebook_file_path], check=True)
    except CalledProcessError as e:
        print(f"Error: {e}")

# Specify the path to your Jupyter Notebook file