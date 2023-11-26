from subprocess import (
    run, CalledProcessError
)

from os import path

from config import paths

def run_analysis(notebook_file_name: str):
    
    notebook_file_path = paths.notebook_folder_path + notebook_file_name
    
    try:
        if not path.exists(notebook_file_path):
            print(f"Error: Notebook file '{notebook_file_path}' not found.")
            return
        
        run(['poetry', 'run', 'jupyter', 'notebook', notebook_file_path], check=True)
    except CalledProcessError as e:
        print(f"Error: {e}")
