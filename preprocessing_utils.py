import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
import requests
import pickle
from ast import literal_eval
from tqdm import tqdm

def read_data_to_df(fn):
    try:
        df = pd.read_csv(fn)
        print("CSV file loaded successfully into a pandas DataFrame.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the file name and path.")
    except pd.errors.EmptyDataError:
        print("No data in the CSV file.")
    except pd.errors.ParserError:
        print("Error while parsing the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_init_df(fp_patient_data):
    df = read_data_to_df(fp_patient_data)

    return df

def format_disease_diagnoses(df):
    df['Diagnoses - secondary ICD10'] = df['Diagnoses - secondary ICD10'].fillna('[]')
    df['Diagnoses - secondary ICD10'] = [np.array(ast.literal_eval(s)) for s in df['Diagnoses - secondary ICD10']]

    df['Diagnoses - ICD10'] = df['Diagnoses - ICD10'].fillna('[]')
    df['Diagnoses - ICD10'] = [np.array(ast.literal_eval(s)) for s in df['Diagnoses - ICD10']]

    def merge_and_remove_duplicates(row):
        return np.unique(np.concatenate((row['Diagnoses - ICD10'], row['Diagnoses - secondary ICD10'])))

    df['Diagnoses'] = df.apply(merge_and_remove_duplicates, axis=1)
    df = df.drop(['Diagnoses - secondary ICD10', 'Diagnoses - ICD10'], axis=1)
    
    return df

def format_lifestyle_demog_data(df, fp_lifestyle_dem):
    lifestyle_demo_df = pd.read_csv(fp_lifestyle_dem)
    lifestyle_demo_df = lifestyle_demo_df.set_index('Participant ID')

    #encoding sex column
    sex_mapping = {
        'Female': 1,
        'Male': 0
    }
    lifestyle_demo_df['Sex'] = lifestyle_demo_df['Sex'].map(sex_mapping)

    #encoding alcohol mapping (ordinal)
    alcohol_mapping = {
        'Prefer not to answer': np.NaN,
        'Never': 0,
        'Once or twice a week' : 1,
        'Three or four times a week' : 2,
        'One to three times a month' : 3,
        'Special occasions only' : 4,
        'Daily or almost daily' : 5
    }
    lifestyle_demo_df['Alcohol intake frequency. | Instance 0'] = lifestyle_demo_df['Alcohol intake frequency. | Instance 0'].map(alcohol_mapping)

    # encode ethnicity
    lifestyle_demo_df['Ethnic background | Instance 0'] = lifestyle_demo_df['Ethnic background | Instance 0'].apply(lambda x: ast.literal_eval(x)[0] if pd.notna(x) else x)

    lifestyle_demo_df['Ethnic background | Instance 0'].replace('Any other mixed background', 'Mixed', inplace=True)
    lifestyle_demo_df['Ethnic background | Instance 0'].replace('Any other white background', 'White', inplace=True)
    lifestyle_demo_df['Ethnic background | Instance 0'].replace('Do not know', np.NaN, inplace=True)
    lifestyle_demo_df['Ethnic background | Instance 0'].replace('Prefer not to answer', np.NaN, inplace=True)
    lifestyle_demo_df['Ethnic background | Instance 0'].replace('Unknown', np.NaN, inplace=True)

    # Fill NaN values with a placeholder string
    lifestyle_demo_df['Ethnic background | Instance 0'].fillna('Unknown', inplace=True)

    # Initialize the label encoder
    le = LabelEncoder()

    # Fit and transform the 'ethnicity' column
    lifestyle_demo_df['Ethnic background | Instance 0'] = le.fit_transform(lifestyle_demo_df['Ethnic background | Instance 0'])
    # Display the ethnicity mapping (optional)
    #for original, encoded in zip(le.classes_, range(len(le.classes_))):
    #    print(f"'{original}' is encoded as {encoded}")

    # ---TRANSFORMATIONS---
    # dividing Pack years of smoking | Instance 0 with age at recruitment | Instance 0 to get the number of years smoked
    lifestyle_demo_df['Lifetime Smoking Proportion'] = lifestyle_demo_df['Pack years of smoking | Instance 0'] / lifestyle_demo_df['Age at recruitment']
    lifestyle_demo_df = lifestyle_demo_df.drop(['Pack years of smoking | Instance 0'], axis=1)

    # if ever smoked is 0, then pack years of smoking is 0
    lifestyle_demo_df.loc[lifestyle_demo_df['Ever smoked | Instance 0'] == 'No', 'Lifetime Smoking Proportion'] = 0
    lifestyle_demo_df = lifestyle_demo_df.drop(['Ever smoked | Instance 0'], axis=1)

    # convert duration of moderate activity to hours per day
    lifestyle_demo_df['Duration of moderate activity | Instance 0'] = pd.to_numeric(lifestyle_demo_df['Duration of moderate activity | Instance 0'], errors='coerce')
    lifestyle_demo_df['Duration of moderate activity | Instance 0'] = lifestyle_demo_df['Duration of moderate activity | Instance 0'] / 60

    #convert to categorical
    lifestyle_demo_df['Sex'] = lifestyle_demo_df['Sex'].astype(pd.CategoricalDtype(categories=np.unique(lifestyle_demo_df['Sex']), ordered=False))
    lifestyle_demo_df['Ethnic background | Instance 0'] = lifestyle_demo_df['Ethnic background | Instance 0'].astype(pd.CategoricalDtype(categories=np.unique(lifestyle_demo_df['Ethnic background | Instance 0']), ordered=True))
    # remove rows with 'Sleep duration | Instance 0' == 'Do not know'
    lifestyle_demo_df = lifestyle_demo_df[lifestyle_demo_df['Sleep duration | Instance 0'] != 'Do not know']
    lifestyle_demo_df = lifestyle_demo_df[lifestyle_demo_df['Sleep duration | Instance 0'] != 'Prefer not to answer']
    lifestyle_demo_df['Sleep duration | Instance 0'].fillna('missing', inplace=True)
    unique_values = np.unique(lifestyle_demo_df['Sleep duration | Instance 0'])
    lifestyle_demo_df['Sleep duration | Instance 0'] = lifestyle_demo_df['Sleep duration | Instance 0'].astype(pd.CategoricalDtype(categories=unique_values, ordered=True))

    # merge df with lifestyle_demo_df
    df = df.join(lifestyle_demo_df, how='left')

    return df

def format_biomarkers_data(df):
    # Create an empty dictionary to hold the DataFrames
    olink_dfs = {}

    # Loop through the file numbers to read each CSV into a DataFrame
    for i in range(1, 7):  
        file_name = f'biomarker_data/olink_df{i}.csv'
        olink_dfs[f'olink_df{i}'] = pd.read_csv(file_name)

    # merging dataframes wrt 'eid'
    biomarker_df = olink_dfs['olink_df1']

    # Loop through the remaining DataFrames in the dictionary and merge them one by one
    for i in range(2, 7):
        biomarker_df = pd.merge(biomarker_df, olink_dfs[f'olink_df{i}'], on='eid', how='outer')

    print(f'Shape of biomarkers data:{biomarker_df.shape}')

    # looking at overlapping patients between df and biomarker_df
    overlapping_indices = df.index[df.index.isin(biomarker_df['eid'])]
    print(f"We have {len(overlapping_indices.tolist())} overlapping indices.")

    # convert gene column names to all caps
    current_columns = biomarker_df.columns.tolist()
    current_columns[:-13] = [col.upper() for col in current_columns[:-13]]
    biomarker_df.columns = current_columns

    # merging biomarker_df with df
    enhanced_df = pd.merge(biomarker_df, df, left_on='EID', right_index=True, how='inner')
    
    return enhanced_df

def format_drug_names_DB(df):
    def safe_literal_eval(val):
        try:
            evaluated_val = ast.literal_eval(val)

            # Check if the evaluated value is a string, then convert to lowercase
            if isinstance(evaluated_val, str):
                return evaluated_val.lower()
            else:
                return evaluated_val
                
        except (ValueError, SyntaxError):
            return np.nan
    
    def to_lower(val):
        if isinstance(val, str):
            return val.lower()
        else:
            return val

    # Function to replace NaN values with an empty list
    def replace_nan_with_list(x):
        if isinstance(x, (list, np.ndarray)):
            return x
        if pd.isna(x):
            return []
        return x

    # Function to replace real names with encoded names in a listd
    def replace_names(drug_list):
        if drug_list is None:
            return None
        elif isinstance(drug_list, (float, int)):  # Handle numerical types, which may include NaN
            return drug_list
        elif isinstance(drug_list, list):  # Assuming the column is a list of strings
            return [encoding_map[drug] for drug in drug_list if drug in encoding_map]
        else:
            return drug_list  # Fallback for other data types
    ##########################################################

    # Convert the string representation of a list to a list
    df['Treatment/medication code | Instance 0'] = df['Treatment/medication code | Instance 0'].apply(safe_literal_eval)
    
    # Read the drug_names.tsv file
    db_vocab = pd.read_csv('kg_data/drugbank vocabulary.csv')
    db_vocab = db_vocab[['DrugBank ID', 'Common name','Synonyms']]
    db_vocab['Common name'] = db_vocab['Common name'].apply(to_lower)
    db_vocab['Synonyms'] = db_vocab['Synonyms'].apply(to_lower)
    db_vocab['Synonyms'] = db_vocab['Synonyms'].apply(lambda x: x.split(' | ') if isinstance(x, str) else x)
    db_vocab = db_vocab.fillna("[]")

    # Initialize an empty dictionary to hold the mappings
    encoding_map = {}

    # Iterate through the DataFrame rows
    for idx, row in db_vocab.iterrows():
        # Add mapping from Col1 to Col2
        encoding_map[row['Common name']] = row['DrugBank ID']

        # Add mappings from elements in Col3 to Col2
        if row['Synonyms']:
            for item in row['Synonyms']:
                encoding_map[item] = row['DrugBank ID']

    # Create a dictionary that maps REAL_NAME to ENCODED_NAME
    #name_dict = db_vocab.set_index('Common name')['DrugBank ID'].to_dict()
    
    # Apply the function to the column of interest
    df['Treatment/medication code | Instance 0'] = df['Treatment/medication code | Instance 0'].apply(replace_names)
    
    # Replace NaN values with an empty list
    df['Treatment/medication code | Instance 0'] = df['Treatment/medication code | Instance 0'].apply(replace_nan_with_list)
    #df['Treatment/medication code | Instance 0'] = df['Treatment/medication code | Instance 0'].apply(ast.literal_eval)
    
    return df

# This wrapper function saves DataFrame to a pickle file
def save_progress(df, filename='outputs/progress.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    
def fetch_umls_cuis(apikey, version='current', sabs=None, ttys=None, string_list=None, verbose=False):
    base_uri = 'https://uts-ws.nlm.nih.gov'
    
    # Initialize a list to hold rows
    rows_list = []
    
    # Initialize an empty list if string_list is None
    if string_list is None:
        string_list = []

    for string in tqdm(string_list):
        page = 0
        
        while True:
            page += 1
            path = f'/search/{version}'
            query = {
                'string': string,
                'apiKey': apikey,
                'rootSource': sabs,
                'termType': ttys,
                'pageNumber': page
            }
            
            try:
                output = requests.get(f'{base_uri}{path}', params=query)
                output.encoding = 'utf-8'
                outputJson = output.json()
                
                # Check if 'result' and 'results' keys exist in outputJson
                results = outputJson.get('result', {}).get('results')
                
                if results is None:
                    if verbose: print(f'No "result" key found for string: {string}')
                    break
                
            except requests.RequestException as e:
                if verbose: print(f"An error occurred: {e}")
                break

            if len(results) == 0:
                if page == 1:
                    if verbose: print(f'No results found for string: {string}')
                    break
                else:
                    break

            for item in results:
                new_row = {
                    "Search_String": string,
                    "UI": item['ui'],
                    "Name": item['name'],
                    "URI": item['uri'],
                    "Source_Vocabulary": item['rootSource']
                }
                
                rows_list.append(new_row)
                
    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows_list)
    CUIs_list = df['UI'].to_numpy()

    return CUIs_list
    
def format_disease_names_CUI(df, apikey, verbose=False):
    def split_each_string(arr):
        return np.array([x.split(' ', 1)[1] if ' ' in x else x for x in arr])
    
    df['Diagnoses'] = df['Diagnoses'].apply(split_each_string)
    # Convert every string in 'Diagnoses' column to lower case
    df['Diagnoses'] = df['Diagnoses'].apply(lambda x: np.char.lower(x))

    version = 'current'
    sabs = None
    ttys = None

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Diagnoses"):
        if not row['Processed']:  # Only process if the row has not been processed
            element = row['Diagnoses']
            cui_list = fetch_umls_cuis(apikey, version, sabs, ttys, element)
            
            df.at[index, 'CUIs'] = cui_list
            df.at[index, 'Processed'] = True  # Mark the row as processed
            
            # Save progress every 2 iterations
            if index % 5 == 0:
                save_progress(df)
                if verbose: print('--------------------progress saved for index:', index, '--------------------')

    return df

def format_disease_names(df, apikey, verbose=False, first_time=True):
    # Initialize df['Processed'] = False if you're creating df for the first time
    
    if first_time:
        df['Processed'] = False
    
    # Load existing progress if it exists
    try:
        with open('outputs/progress.pkl', 'rb') as f:
            df = pickle.load(f)
    except FileNotFoundError:
        # Your code to initialize df if no saved progress exists
        pass
    df = format_disease_names_CUI(df, apikey, verbose)
    
    # final preprocessing
    # Add commas where needed
    df['CUIs'] = df['CUIs'].str.replace(r"(' ')|(' ')", "', '", regex=True)
    # Remove extra enclosing quotes
    df['CUIs'] = df['CUIs'].str.strip('"')
    # Replace new line character \n with a comma and a space
    df['CUIs'] = df['CUIs'].str.replace('\n ', ', ')
    df['CUIs'] = df['CUIs'].str.strip('"')

    def safe_literal_eval(cell):
        if isinstance(cell, str):
            cell = cell.replace(" ... ", ", ")  # replace the ellipsis with a comma if it's used as a separator
            try:
                return literal_eval(cell)
            except Exception as e:
                print(f"Failed to convert string {cell}")
                print(f"Exception: {e}")
                return cell  # or return None if you prefer
        else:
            return cell  # If it's already a list, no need to do anything

    df['CUIs'] = df['CUIs'].apply(safe_literal_eval)

    save_progress(df, 'final_progress.pkl')
    return df
        