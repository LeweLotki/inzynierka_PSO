import subprocess

notebook_file_path = './data_analysis/notebook.ipynb'

def run_analysis(file_path=notebook_file_path):
    try:
        # Use the subprocess module to run the jupyter notebook command
        subprocess.run(['poetry', 'run', 'jupyter', 'notebook', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Specify the path to your Jupyter Notebook file