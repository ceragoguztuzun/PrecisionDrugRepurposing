import os
import random
from collections import defaultdict
import argparse

''' file I/O functions '''
def write_lines_to_file(lines, filepath):
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(f"{line}\n")

def read_nodes_from_file(filepath):
    nodes = set()
    with open(filepath, 'r') as file:
        for line in file.readlines():
            node1, _, node2 = line.strip().split('\t')
            nodes.add(node1)
            nodes.add(node2)
    return nodes

def collect_nodes(lines):
    nodes = set()
    for line in lines:
        node1, _, node2 = line.strip().split('\t')
        nodes.add(node1)
        nodes.add(node2)
    return list(nodes)


def delete_test_data_files():
    test_data_dir = os.path.join(os.path.dirname(__file__), 'code/data/test_data')
    
    if not os.path.exists(test_data_dir):
        print(f"The directory {test_data_dir} does not exist.")
        return

    for filename in os.listdir(test_data_dir):
        file_path = os.path.join(test_data_dir, filename)
        try:
            if os.path.isfile(file_path):
                if filename != 'ad_pre.txt':  # Skip deleting 'ad_pre.txt'
                    os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

''' split function '''
def split_GPKG(kg_filepath):
    try:
        # Step 1: Read the file
        file_path = os.path.join(os.path.dirname(__file__), kg_filepath)#'data/knowledge_graph.txt')
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        
        total_edges = len(lines)
        print(f"Total edges in the graph: {total_edges}")
        
        # Step 2: Identify all unique nodes
        all_nodes = set()
        for line in lines:
            node1, _, node2 = line.split('\t')
            all_nodes.add(node1)
            all_nodes.add(node2)
            
        print(f"Total unique nodes: {len(all_nodes)}")
        
        # Step 4: Find an edge set that contains all unique nodes at least once
        train_lines = set()
        nodes_in_train = set()
        
        for line in lines:
            node1, _, node2 = line.split('\t')
            if node1 not in nodes_in_train or node2 not in nodes_in_train:
                train_lines.add(line)
                nodes_in_train.add(node1)
                nodes_in_train.add(node2)
                
        print(f"Edges in preliminary training set: {len(train_lines)}")
        print(f"Nodes in preliminary training set: {len(nodes_in_train)}")
        
        # Set desired ratios for Test and Valid sets
        test_ratio = 0.2
        valid_ratio = 0.2
        # The remaining will automatically belong to the training set
        
        remaining_lines = set(lines) - train_lines
        remaining_lines = list(remaining_lines)  # Convert to list for random shuffling
        random.shuffle(remaining_lines)
        
        total_remaining = len(remaining_lines)
        valid_count = int(total_remaining * valid_ratio)
        test_count = int(total_remaining * test_ratio)
        
        valid_lines = remaining_lines[:valid_count]
        test_lines = remaining_lines[valid_count:(valid_count + test_count)]
        train_lines.update(remaining_lines[(valid_count + test_count):])  # Add the rest to the training set

        total_edges = len(lines)
        
        # Report the ratios
        train_ratio = len(train_lines) / total_edges * 100
        valid_ratio = len(valid_lines) / total_edges * 100
        test_ratio = len(test_lines) / total_edges * 100
        
        print(f"Train:Test:Valid ratio = {train_ratio:.2f}%:{test_ratio:.2f}%:{valid_ratio:.2f}%")
        
        # Write to files
        base_output_dir = os.path.join(os.path.dirname(__file__), 'code/data/test_data')
        
        write_lines_to_file(train_lines, os.path.join(base_output_dir, 'train.txt'))
        write_lines_to_file(valid_lines, os.path.join(base_output_dir, 'valid.txt'))
        write_lines_to_file(test_lines, os.path.join(base_output_dir, 'test.txt'))

    except FileNotFoundError:
        print(f"File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


''' testing functions (testing whether all nodes in the validation and test sets also appear in the training set) '''
def test_splits(train_filepath, valid_filepath, test_filepath):
    # Read nodes from each file
    train_nodes = read_nodes_from_file(train_filepath)
    valid_nodes = read_nodes_from_file(valid_filepath)
    test_nodes = read_nodes_from_file(test_filepath)

    # Check if all nodes in valid and test sets are also in the train set
    valid_diff = valid_nodes.difference(train_nodes)
    test_diff = test_nodes.difference(train_nodes)

    if len(valid_diff) == 0 and len(test_diff) == 0:
        print("Test Passed: All nodes in the validation and test sets also appear in the training set.")
    else:
        print("Test Failed")
        if len(valid_diff) > 0:
            print(f"Nodes in validation set not found in training set: {len(valid_diff)}")
        if len(test_diff) > 0:
            print(f"Nodes in test set not found in training set: {len(test_diff)}")

def run_test():
    base_dir = 'code/data/test_data'  # Replace with the path where your txt files are stored
    train_filepath = os.path.join(base_dir, 'train.txt')
    valid_filepath = os.path.join(base_dir, 'valid.txt')
    test_filepath = os.path.join(base_dir, 'test.txt')

    test_splits(train_filepath, valid_filepath, test_filepath)

''' main '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a knowledge graph into training, validation, and test sets.")
    parser.add_argument('-kg_filepath', type=str, default=None, help="Path to the knowledge graph file.")

    args = parser.parse_args()
    kg_filepath = args.kg_filepath

    delete_test_data_files()

    if kg_filepath:
        split_GPKG(kg_filepath)
    else:
        split_GPKG()

    run_test()
