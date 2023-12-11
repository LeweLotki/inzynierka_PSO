from data_extraction.run_extraction import run_extraction
from data_analysis.run_analysis import run_analysis
from data_postprocessing.cost_function import cost_function

from config import paths

from argparse import ArgumentParser

from os import listdir

def default_message():
    
    print("\r No option specified. \n\r Type '--help' for list of arguments.")

def get_description(path: str) -> str:
    
    with open(path, 'r', encoding='utf-8') as file:
        description = file.read().strip()
        return description
    
if __name__ == "__main__":
    
    program_description = get_description(path=paths.description_path)
    data_analysis_description = get_description(path=paths.data_analysis_description_path)
    data_extraction_description = get_description(path=paths.data_extraction_description_path)
    data_postprocessing_description = get_description(path=paths.data_postprocessing_description_path)
    sampling_coef_description = get_description(path=paths.sampling_coef_description_path)
    
    notebook_directory = paths.notebook_folder_path
    available_notebooks = [file for file in listdir(notebook_directory) if file.endswith('.ipynb')]
    
    parser = ArgumentParser(description=program_description)
    
    parser.add_argument('-a', '--notebook', help=data_analysis_description + f' {", ".join(available_notebooks)}')
    parser.add_argument('-e', action='store_true', help=data_extraction_description)
    parser.add_argument('-p', action='store_true', help=data_postprocessing_description)
    parser.add_argument('--sampling_coef', type=float, help=sampling_coef_description)

    args = parser.parse_args()
    
    if args.notebook: run_analysis(args.notebook)
    elif args.e: run_extraction()
    elif args.p: 
        if args.sampling_coef:
            cost_function(sampling_coef=args.sampling_coef)
        else: cost_function()
    else: default_message()
