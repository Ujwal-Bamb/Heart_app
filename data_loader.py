
#data_loader.py
# Import the pandas library for working with data
import pandas as pd

# Define a function to load the heart.csv file
def load_data(path='data/heart.csv'):

    try:
        # Try to read the CSV file from the given path
        df = pd.read_csv(path)

        # If successful, print a confirmation message
        print(" Data loaded successfully!")

        # Return the loaded DataFrame (table of data)
        return df

    # If the file is not found at the given path, show an error
    except FileNotFoundError:
        print(f"❌ File not found at: {path}")