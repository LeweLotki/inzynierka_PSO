from data_extraction.run_extraction import run_extraction
from data_analysis.run_analysis import run_analysis

from argparse import ArgumentParser

from os import listdir

def default_message():
    
    print("\r No option specified. \n\r Type '--help' for list of arguments.")

if __name__ == "__main__":
    
    description_path = './attributes/description.txt'
    with open(description_path, 'r') as file:
        program_description = file.read().strip()
    
    notebook_directory = './data_analysis/notebooks'
    available_notebooks = [file for file in listdir(notebook_directory) if file.endswith('.ipynb')]
    
    parser = ArgumentParser(description=program_description)

    # Define command-line options
    parser.add_argument('-a', '--notebook', help=f'Specify the Jupyter Notebook file containing data analysis. '
                                                 f'Possible values: {", ".join(available_notebooks)}')
    parser.add_argument('-e', action='store_true', help='extracting features from photos of particles and writes the results to CSV files.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Check which option was provided and execute the corresponding function
    if args.notebook: run_analysis(args.notebook)
    elif args.e: run_extraction()
    else: default_message()
