import pandas as pd
import numpy as np
import ast
import re
from tqdm import tqdm
import torch
import preprocessing_utils as pp_utils
from ast import literal_eval
import argparse

class Patient_Merge:

    patient_id = None
    patient_df = None

    def __init__(self, kg_fp, patients_df_fp, verbose = False):
        
        self.verbose = verbose

        self.kg_df = self.read_KG(kg_fp)
        self.patients_df = self.read_patients_df(patients_df_fp)

        #get unique values of Head and Tail
        self.unique_heads = self.kg_df['Head'].unique()
        self.unique_tails = self.kg_df['Tail'].unique()

    @classmethod
    def read_KG(self, fp):
        # Read the text file and create the list of tuples
        triples = []
        with open(fp, "r") as f:
            for line in f:
                elements = line.strip().split("\t", maxsplit=2)
                if len(elements) == 3:
                    head, edge, tail = elements
                    triples.append((head, edge, tail))
                else:
                    if self.verbose: print(f"Warning: Skipping line '{line.strip()}' as it does not have exactly 3 elements separated by tabs.")

        # Convert the list of tuples to a DataFrame
        kg_df = pd.DataFrame(triples, columns=["Head", "Relation", "Tail"])
        return kg_df
    
    @classmethod
    def read_patients_df(self, fp):
        patients_df = pd.read_csv(fp)

        return patients_df

    def pick_patient(self, patient_id):
        #pick a patient
        patient = self.patients_df[self.patients_df['EID'] == patient_id]
        Patient_Merge.patient_id = int(patient_id)
        Patient_Merge.patient_df = patient
        #self.format_CUI()
        #for c in self.patient_df['CUIs']:
        #    print(c)
        return patient
    
    def get_kg(self):
        return self.kg_df

    def merge_nodes(self):
        self.filter_and_match_drug_nodes()
        self.filter_and_match_disease_nodes_PRS()
        self.filter_and_match_disease_nodes_PRS_comorbities()
        self.filter_and_match_disease_nodes_DisHist()
        self.filter_and_match_gene_nodes_Biomarkers()

    def filter_and_match_drug_nodes(self):
        # Explicitly create a copy of self.patient_df to avoid SettingWithCopyWarning
        self.patient_df = self.patient_df.copy()

        # Rest of your code remains the same
        db_heads = [head for head in self.unique_heads if head.startswith('DB')]
        db_tails = [tail for tail in self.unique_tails if tail.startswith('DB')]

        db_entities = db_heads + db_tails
        drugs_in_KG = set(db_entities)

        if self.verbose: print(f'There are {len(drugs_in_KG)} unique drugs in the KG')

        self.patient_df.loc[:, 'Treatment/medication code | Instance 0'] = self.patient_df['Treatment/medication code | Instance 0'].apply(ast.literal_eval)

        def filter_drugs(drug_list):
            return [drug for drug in drug_list if drug in drugs_in_KG]

        self.patient_df['Filtered Treatment/medication'] = self.patient_df['Treatment/medication code | Instance 0'].apply(filter_drugs)

        # Uncomment and adapt this code if needed for creating self-loop
        '''
        for drug in filtered_patients_drugs_list:
            new_relation = {'Head': drug, 'Relation': 'HAS_USED', 'Tail': drug}
            self.kg_df.loc[len(self.kg_df.index)] = new_relation
            if self.verbose: print('Added self-loop for drug:', drug)
        '''

    def filter_and_match_disease_nodes_PRS(self): 
        
        # get elements in unique_heads that start with 'C'
        disease_heads = list(filter(lambda head: re.match(r'^C\d+$', head), self.unique_heads))
        disease_tails = list(filter(lambda tail: re.match(r'^C\d+$', tail), self.unique_tails))

        #merge the two lists
        disease_entities = disease_heads + disease_tails
        diseases_in_KG = set(disease_entities)

        if self.verbose: print(F'There are {len(diseases_in_KG)} unique diseases in the KG')

        # get CUI's of Alzheimer’s disease
        UMLS_apikey = "68297838-4f73-4972-b664-b5484a59b087"

        CUI_list_AD = pp_utils.fetch_umls_cuis(UMLS_apikey, version='current', sabs=None, ttys=None, string_list=['Alzheimer\'s Disease'], verbose=True)
        CUI_list_AD = np.unique(CUI_list_AD)

        if self.verbose: print(F'There are {len(CUI_list_AD)} unique CUIs for AD from UMLS.')

        # filter patient data
        filtered_AD_CUI_list = [item for item in CUI_list_AD if item in diseases_in_KG]
        if self.verbose: print(F'There are {len(filtered_AD_CUI_list)} overlapping CUIs for AD from UMLS and KG.')

        # create self-loop
        for AD_CUI in filtered_AD_CUI_list:
            scalar = self.patient_df['Standard PRS for alzheimer\'s disease (AD)'].values[0]
            new_relation = {'Head': AD_CUI, 'Relation': scalar, 'Tail': AD_CUI}
            self.kg_df.loc[len(self.kg_df.index)] = new_relation
    
    def filter_and_match_disease_nodes_PRS_comorbities(self):
        # get diseases in KG
        # get elements in unique_heads that start with 'C'
        disease_heads = list(filter(lambda head: re.match(r'^C\d+$', head), self.unique_heads))
        disease_tails = list(filter(lambda tail: re.match(r'^C\d+$', tail), self.unique_tails))
        #merge the two lists
        disease_entities = disease_heads + disease_tails
        diseases_in_KG = set(disease_entities)

        #for each PRS, check if it is in the KG
        prs_cols = [col for col in self.patient_df.columns if col.startswith('PRS_')]
        
        for col in prs_cols:
            #remove PRS_ from col
            col_cui = col[4:]
            #check if col is in diseases_in_KG
            if col_cui in diseases_in_KG:
                #create self-loop
                scalar = self.patient_df[col].values[0]
                new_relation = {'Head': col, 'Relation': scalar, 'Tail': col}
                self.kg_df.loc[len(self.kg_df.index)] = new_relation
    
    def filter_and_match_disease_nodes_DisHist(self):
        # get elements in unique_heads that start with 'DB'
        disease_heads = list(filter(lambda head: re.match(r'^C\d+$', head), self.unique_heads))
        disease_tails = list(filter(lambda tail: re.match(r'^C\d+$', tail), self.unique_tails))

        #merge the two lists
        disease_entities = disease_heads + disease_tails
        diseases_in_KG = set(disease_entities)

        # filter patient data
        patient_CUIs = self.format_CUI(self.patient_df['CUIs'].values[0])

        filtered_DH_CUI_list = []
        for item in np.unique(patient_CUIs):
            if item in diseases_in_KG:
                filtered_DH_CUI_list.append(item)

        if self.verbose: print(f'There are {len(filtered_DH_CUI_list)} overlapping CUIs from Disease History of patient and KG.')

        # create self-loop
        for dishist_CUI in filtered_DH_CUI_list:
            new_relation = {'Head': dishist_CUI, 'Relation': 'HAS_DIAGNOSED', 'Tail': dishist_CUI}
            self.kg_df.loc[len(self.kg_df.index)] = new_relation

    def filter_and_match_gene_nodes_Biomarkers(self):
        # Filter rows based on condition
        condition = (
            (self.kg_df['Relation'] == 'TARGET') | 
            (self.kg_df['Relation'] == 'ASSOCIATE') | 
            self.kg_df['Relation'].str.startswith('RELATE')
        )
        filtered_kg_df = self.kg_df[condition][['Head', 'Tail']]
        
        #get unique values of Head and Tail
        unique_heads_filtered_kg = filtered_kg_df['Head'].unique()
        unique_tails_filtered_kg = filtered_kg_df['Tail'].unique()

        #merge the two lists
        entities_filtered_kg = np.concatenate((unique_tails_filtered_kg, unique_heads_filtered_kg))
        entities_filtered_kg = set(entities_filtered_kg)
        
        # remove biomarker columns from patient data if it is not in the KG
        #get columns of patient_1
        patient_1_columns = self.patient_df.columns.tolist()
        patient_1_columns = patient_1_columns[1:-14]
        
        # get patient_1_columns that are in entities_filtered_kg
        patient_1_columns_to_keep = [col for col in patient_1_columns if col in entities_filtered_kg]

        if self.verbose: print(F'There are {len(patient_1_columns_to_keep)} to keep in patient_1_columns.')

        # add self loop for biomarkers
        for gene in tqdm(patient_1_columns_to_keep, desc='Adding biomarker self-loops'):
            new_relation = {'Head': gene, 'Relation': self.patient_df[gene].values[0], 'Tail': gene}
            self.kg_df.loc[len(self.kg_df.index)] = new_relation

    def save(self, save_scalars=False, save_kg=False):
        scalars = torch.ones(self.kg_df.shape[0], dtype=torch.float)

        # Loop through the dataframe to update the tensor
        for i, value in enumerate(self.kg_df['Relation']):
            if isinstance(value, (int, float)):
                scalars[i] = value

        # WRITE SCALARS AS FILE
        if save_scalars:
            np.save(f'outputs/scalars_{int(self.patient_id)}.npy', scalars.numpy())
            print(f'-> Scalars saved to outputs/scalars_{int(self.patient_id)}.npy')

        # WRITE KG AS FILE
        if save_kg:
            # Replace numeric values in 'Relation' with 'SELF' before saving
            self.kg_df['Relation'] = self.kg_df['Relation'].apply(lambda x: 'SELF' if isinstance(x, (int, float)) else x)
            self.kg_df.to_csv(f'outputs/patient_merged_kg_{int(self.patient_id)}.txt', sep='\t', header=False, index=False)
            print(f'-> Patient-merged KG is saved to outputs/patient_merged_kg_{int(self.patient_id)}.txt')

    def format_CUI(self, cui_list):
        def safe_literal_eval(cell):
            if isinstance(cell, str):
                cell = cell.replace(" ", ", ")  # replace the space with a comma if it's used as a separator
                cell = cell.replace(" ... ", ", ")  # replace the ellipsis with a comma if it's used as a separator
                try:
                    evaluated = literal_eval(cell)
                    # Remove Ellipsis if it's present
                    return [item for item in evaluated if item != Ellipsis]
                except Exception as e:
                    print(f"Failed to convert string {cell}")
                    print(f"Exception: {e}")
                    return cell  # or return None if you prefer
            else:
                return cell  # If it's already a list, no need to do anything

        cui_list = safe_literal_eval(cui_list)

        return cui_list

def main(args):

    patient_merge = Patient_Merge(kg_fp = args.kg_filepath, patients_df_fp = args.patients_data_filepath, verbose = True)
    kg = patient_merge.get_kg()
    chosen_patient = patient_merge.pick_patient(patient_id = args.patient_id)
    
    print(f'-> Merging patient with ID: {patient_merge.patient_id}...')

    #for each EID in the dataframe, get the corresponding patient from the knowledge graph
    patient_merge.merge_nodes()
    
    # save
    patient_merge.save(save_scalars=True, save_kg=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-kg_fp', '--kg_filepath', required=True, type=str, help='The filepath that has the txt file of the knowledge graph.')
    parser.add_argument('-patients_data_fp', '--patients_data_filepath', required=True, type=str, help='The filepath that has the csv file of the patients data.')
    parser.add_argument('-pid', '--patient_id', required=True, type=int, help='The id of the patient of interest, must be present under a "EID" column in patient data csv')
    
    args = parser.parse_args()
    main(args)