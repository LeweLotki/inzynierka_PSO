from data_extraction.run_extraction import run_extraction
from data_analysis.run_analysis import run_analysis
import argparse

def default_message():
    
    print("\r No option specified. \n\r Type '--help' for list of arguments.")

if __name__ == "__main__":
    
    description_path = './attributes/description.txt'
    with open(description_path, 'r') as file:
        program_description = file.read().strip()
    
    parser = argparse.ArgumentParser(description=program_description)

    # Define command-line options
    parser.add_argument('-a', action='store_true', help='data analysis')
    parser.add_argument('-e', action='store_true', help='data extraction')

    # Parse command-line arguments
    args = parser.parse_args()

    # Check which option was provided and execute the corresponding function
    if args.a: run_analysis()
    elif args.e: run_extraction()
    else: default_message()
