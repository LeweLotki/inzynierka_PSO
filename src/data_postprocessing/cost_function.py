from os import (
    listdir, path
)

from ast import literal_eval

from pandas import (
    DataFrame, read_csv, concat
)

from scipy.stats import gaussian_kde

from numpy import (
    meshgrid, linspace, vstack
)

from config import paths

class cost_function:
    
    df = DataFrame()
    
    def __init__(
            self,
            sampling_coef = 5e2, 
            folder_path=paths.csv_folder_path, 
            filter={
                'low':4, 
                'high':None
                }
        ):
        
        self.folder_path = folder_path
        self.filter = filter
        self.sampling_coef = sampling_coef
        
        self.__combine_data()
        self.__filter_data()
        self.__get_cost_function()
        self.__write_cost_function()

    def __combine_data(self):
        
        csv_files = [file for file in listdir(self.folder_path) if file.endswith('.csv')  and "[" in file]
        
        combined_df = DataFrame()
        
        for csv_file in csv_files:
            file_path = path.join(self.folder_path, csv_file)
            temp_df = read_csv(file_path)
            combined_df = concat([combined_df, temp_df], ignore_index=True)
        
        self.df = combined_df

    def __filter_data(self):
        
        if self.filter['low']:
            self.df = self.df[self.df['area'] > self.filter['low']]
        if self.filter['high']:
            self.df = self.df[self.df['area'] < self.filter['high']]

    def __get_cost_function(self):

        self.df[['x', 'y']] = DataFrame(self.df['centroid'].apply(literal_eval).tolist(), index=self.df.index)
        self.df = self.df[['x', 'y', 'area']].dropna()
        self.df[['x', 'y', 'area']] = self.df[['x', 'y', 'area']].astype(float)

        kde = gaussian_kde(self.df[['x', 'y']].T, weights=self.df['area'])

        xv, yv = meshgrid(linspace(self.df['x'].min(), self.df['x'].max(), int(self.sampling_coef)),
                            linspace(self.df['y'].min(), self.df['y'].max(), int(self.sampling_coef)))

        zv = kde(vstack([xv.ravel(), yv.ravel()]))
        zv = zv.reshape(xv.shape)

        self.df = DataFrame({'x': xv.ravel(), 'y': yv.ravel(), 'density': zv.ravel()})
        
    def __write_cost_function(self):
        
        
        # def get_next_filename(folder_path, base_filename):
        # Get a list of all files in the folder
        folder_path = paths.cost_functions_folder_path
        files = [f for f in listdir(folder_path) if path.isfile(path.join(folder_path, f))]

        # Count the number of existing CSV files in the folder
        csv_files = [f for f in files if f.endswith('.csv')]
        num_existing_files = len(csv_files)

        # Determine the next file number
        if num_existing_files == 0:
            next_file_number = ''
        else:
            next_file_number = str(num_existing_files + 1)

        # Construct the new filename
        base_filename = 'cost_function'
        new_filename = f"{base_filename}{next_file_number}.csv"
        
        self.df.to_csv(new_filename, index=False)