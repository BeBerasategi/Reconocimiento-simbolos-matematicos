"""
Tools for the HASY dataset.

Original file: https://github.com/MartinThoma/HASY/blob/master/hasy/hasy_tools.py
Or: https://pypi.org/project/hasy/

Some changes where made by Beñat Berasategui to update the obsolete code.

To access this module:
import sys
sys.path.append('work/databases/HASY/')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.ndimage
from PIL import Image, ImageDraw
import csv
import os
import random
import pickle

import itertools

# Check "Generate_equiv_dictionary.ipynb"

equiv_dict={
    0 : (r'\mid', r'|'),
    1 : (r'\triangle', r'\Delta', r'\vartriangle'),
    2 : (r'\checked', r'\checkmark'),
    3 : (r'\shortrightarrow', r'\rightarrow', r'\longrightarrow'),
    4 : (r'\Longrightarrow', r'\Rightarrow'),
    5 : (r'\backslash', r'\setminus'),
    6 : (r'\O', r'\o', r'\emptyset', r'\diameter', r'\varnothing'),
    7 : (r'\with', r'\&'),
    8 : (r'\triangledown', r'\nabla'),
    9 : (r'\longmapsto', r'\mapsto'),
    10 : (r'\dotsc', r'\dots'),
    11 : (r'\fullmoon', r'\circ', r'o', '\degree'),
    12 : (r'\varpropto', r'\propto', r'\alpha'),
    13 : (r'\mathsection', r'\S'),
    14 : (r'\vDash', r'\models'),
    15 : (r'c', r'C', r'\subset', r'\mathcal{C}'),
    16 : (r'v', r'V', r'\vee'),
    17 : (r'x', r'X', r'\times'),
    18 : (r'\mathbb{Z}', r'\mathds{Z}'),
    19 : (r'T', r'\top'),
    20 : (r's', r'S', r'\mathcal{S}'),
    21 : (r'z', r'Z', r'\mathcal{Z}'),
    22 : (r'\mathbb{R}', r'\mathds{R}'),
    23 : (r'\mathbb{Q}', r'\mathds{Q}'),
    24 : (r'\mathbb{N}', r'\mathds{N}'),
    25 : (r'\oiint', r'\varoiint'),
    26 : (r'\lhd', r'\triangleleft'),
    27 : (r'\sum', r'\Sigma'),
    28 : (r'\prod', r'\Pi', r'\sqcap'),
    29 : (r'\mathcal{B}', r'B'),
    30 : (r'\mathcal{D}', r'D', r'\mathscr{D}'),
    31 : (r'\mathcal{H}', r'H'),
    32 : (r'\mathcal{M}', r'M'),
    33 : (r'\mathcal{N}', r'N', r'\mathscr{N}'),
    34 : (r'\mathcal{O}', r'O', r'0'),
    35 : (r'\mathcal{P}', r'P'),
    36 : (r'\mathcal{R}', r'R', r'\mathscr{R}'),
    37 : (r'\coprod', r'\amalg', r'\sqcup'),
    38 : (r'\bot', r'\perp'),
    39 : (r'\|', r'\parallel'),
    40 : (r'\ohm', r'\Omega'),
    41 : (r'\#', r'\sharp'),
    42 : (r'\mathcal{A}', r'\mathscr{A}'),
    43 : (r'\epsilon', r'\varepsilon', r'\in', r'\mathcal{E}'),
    44 : (r'\Lambda', r'\wedge'),
    45 : (r'\Leftrightarrow', r'\Longleftrightarrow'),
    46 : (r'\mathds{1}', r'\mathbb{1}'),
    47 : (r'\mathscr{L}', r'\mathcal{L}'),
    48 : (r'\rho', r'\varrho'),
    49 : (r'\odot', r'\astrosun'),
    50 : (r'\cdot', r'\bullet'),
    51 : (r'\chi', r'\mathcal{X}'),
    52 : (r'\beta', r'\ss'),
    53 : (r'\male', r'\mars'),
    54 : (r'\female', r'\venus'),
    55 : (r'\bowtie', r'\Bowtie'),
    56 : (r'\mathcal{T}', r'\tau'),
    57 : (r'\diamond', r'\diamondsuit', r'\lozenge'),
}

def MER(y_true, y_pred, equiv_dict=equiv_dict):
    '''
    By Beñat Berasategui
    
    Arguments
    ---------
    y_true : symbol_id must be provided as str.
    y_pred : symbol_id must be provided as str.
    '''
    # Convert 
    symbolid2latex = get_symbolid2latex()
    y_pred = [symbolid2latex[i] for i in y_pred]
    y_true = [symbolid2latex[i] for i in y_true]
    
    correct = 0
    
    # List all the symbols in the equivalence dictionary:
    equiv_list = list(itertools.chain.from_iterable(equiv_dict.values()))
    
    # Invert the dictionary to find equivalent symbols easier:
    inverted_dict = {value: key for key, values in equiv_dict.items() for value in values}
    
    for i in range(len(y_true)):
        if y_true[i] in equiv_list:
            key = inverted_dict[y_true[i]]
            if y_pred[i] in equiv_dict[key]:
                correct += 1
        else:
            if y_pred[i] == y_true[i]:
                correct += 1
    return correct / len(y_true)

def MER_test():
    vec_true = np.array(['82','88','152','181'])
    vec_pred = np.array(['194','116','959','1103'])
    print("The following number should be 1: ", MER(vec_true, vec_pred))


def load_dataframes(path='/home/jovyan/work/databases/HASY/benat-data/pandas_df/'):
    """
    Load data directly to pandas dataframes, and provide symbol_id - index conversion.

    Parameters
    ----------
    path : str. location of csv files.

    Returns
    -------
    X_train : pandas df. training data
    y_train : pandas df. training data labels   
    X_test  : pandas df. test data
    y_test  : pandas df. test data labels
    list_of_dicts : [symbol_id2index_train, index2symbol_id_train, symbol_id2index_test, index2symbol_id_test]
    """
    
    # Load dicts
    with open(path+'dict_list.pkl', 'rb') as f:
        list_of_dicts = pickle.load(f)
    # list_of_dicts = [symbol_id2index_train, index2symbol_id_train, symbol_id2index_test, index2symbol_id_test]
    
    X_train = pd.read_csv(path+'X_train.csv', index_col=0)
    y_train = pd.read_csv(path+'y_train.csv', index_col=0)
    X_test = pd.read_csv(path+'X_test.csv', index_col=0)
    y_test = pd.read_csv(path+'y_test.csv', index_col=0)
    
    return X_train, y_train, X_test, y_test, list_of_dicts

def load_database(csv_filepath):
    """
    Load a CSV file, convert it into more usable data arrays, and provide symbol_id - index conversion.

    Parameters
    ----------
    csv_filepath : str
        Path to a CSV file

    Returns
    -------
    data : arrays of data
    y : label of each instance, each data array. It uses the indexes by default.
    symbol_id2index : dict that maps a symbol_id as in test.csv and
        train.csv to an integer in 0...k, where k is the total
        number of unique labels.
    symbol_id2index : dict containing indexes as keys and the corresponding symbol_id as value.
    """
    symbol_id2index = generate_index(csv_filepath)
    index2symbol_id = {}
    for index, symbol_id in symbol_id2index.items():
        index2symbol_id[symbol_id] = index
    data, y = load_images(csv_filepath, symbol_id2index, one_hot=False, flatten = True)
    return data, y, symbol_id2index, index2symbol_id

def load_csv(filepath, delimiter=',', quotechar="'"):
    """
    Load a CSV file.

    Parameters
    ----------
    filepath : str
        Path to a CSV file
    delimiter : str, optional
    quotechar : str, optional

    Returns
    -------
    list of dicts : Each line of the CSV file is one element of the list.
    """
    data = []
    csv_dir = os.path.dirname(filepath)
    with open(filepath, 'rt') as csvfile:
        reader = csv.DictReader(csvfile,
                                delimiter=delimiter,
                                quotechar=quotechar)
        for row in reader:
            if 'path' in row:
                row['path'] = os.path.abspath(os.path.join(csv_dir,
                                                           row['path']))
            data.append(row)
    return data

def load_images(csv_filepath, symbol_id2index, one_hot=True, flatten=False):
    """
    Load the images into a 4D uint8 numpy array [index, y, x, depth].

    Parameters
    ----------
    csv_filepath : str
        'test.csv' or 'train.csv'
    symbol_id2index : dict
        Dictionary generated by generate_index
    one_hot : bool, optional
        Make label vector as 1-hot encoding, otherwise index
    flatten : bool, optional
        Flatten feature vector

    Returns
    -------
    images, labels : Images is a 4D uint8 numpy array [index, y, x, depth]
                     and labels is a 2D uint8 numpy array [index][1-hot enc].
    """
    WIDTH, HEIGHT = 32, 32
    dataset_path = os.path.dirname(csv_filepath)  # Main directory of HASY
    data = load_csv(csv_filepath)
    if flatten:
        images = np.zeros((len(data), WIDTH * HEIGHT))
    else:
        images = np.zeros((len(data), WIDTH, HEIGHT, 1))
        #print(np.shape(images)) -> (168233, 32, 32, 1)
    labels = []
    for i, data_item in enumerate(data):
        fname = os.path.join(dataset_path, data_item['path'])
        # img = Image.open(fname).convert('L')
        #print(np.shape(np.asarray(img))) -> (32, 32)

        if flatten:
            img = Image.open(fname).convert('L')
            images[i, :] = np.asarray(img).flatten()
        else:
            img = Image.open(fname).convert('L')
            images[i, :, :, 0] = np.asarray(img)
            
        label = symbol_id2index[data_item['symbol_id']]
        labels.append(label)
    data = images, np.array(labels)
    if one_hot:
        data = (data[0], np.eye(len(symbol_id2index))[data[1]])
    return data


def create_random_overview(img_src, x_images, y_images):
    """Create a random overview of images."""
    # Create canvas
    background = Image.new('RGB',
                           (35 * x_images, 35 * y_images),
                           (255, 255, 255))
    bg_w, bg_h = background.size
    # Paste image on canvas
    for x in range(x_images):
        for y in range(y_images):
            path = random.choice(img_src)['path']
            img = Image.open(path, 'r')
            img_w, img_h = img.size
            offset = (35 * x, 35 * y)
            background.paste(img, offset)
    # Draw lines
    draw = ImageDraw.Draw(background)
    for y in range(y_images):  # horizontal lines
        draw.line((0, 35 * y - 2, 35 * x_images, 35 * y - 2), fill='gray')
    for x in range(x_images):  # vertical lines
        draw.line((35 * x - 2, 0, 35 * x - 2, 35 * y_images), fill='gray')
    # Write fill=0 for black lines    
    
    display(background)
    # Store
    background.save('hasy-overview_1.png')


def get_symbolid2latex(csv_filepath='/home/jovyan/work/databases/HASY/symbols.csv'):
    """Return a dict mapping symbol_ids to LaTeX code."""
    symbol_data = load_csv(csv_filepath)
    symbolid2latex = {}
    for row in symbol_data:
        symbolid2latex[row['symbol_id']] = row['latex']
    return symbolid2latex

def generate_index(csv_filepath):
    """
    Generate an index 0...k for the k labels.

    Parameters
    ----------
    csv_filepath : str
        Path to 'test.csv' or 'train.csv'

    Returns
    -------
    dict : Maps a symbol_id as in test.csv and
        train.csv to an integer in 0...k, where k is the total
        number of unique labels.
    """
    symbol_id2index = {}
    data = load_csv(csv_filepath)
    i = 0
    for item in data:
        if item['symbol_id'] not in symbol_id2index:
            symbol_id2index[item['symbol_id']] = i
            i += 1
    return symbol_id2index

def _analyze_class_distribution(csv_filepath,
                                max_data=1000,
                                bin_size=25):
    """Plot the distribution of training data over graphs."""
    symbol_id2index = generate_index(csv_filepath)
    index2symbol_id = {}
    for index, symbol_id in symbol_id2index.items():
        index2symbol_id[symbol_id] = index
    data, y = load_images(csv_filepath, symbol_id2index, one_hot=False)

    data = {}
    for el in y:
        if el in data:
            data[el] += 1
        else:
            data[el] = 1
    classes = data
    images = len(y)

    # Create plot
    print("Classes: %i" % len(classes))
    print("Images: %i" % images)

    class_counts = sorted([count for _, count in classes.items()])
    print("\tmin: %i" % min(class_counts))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # plt.title('HASY training data distribution')
    plt.xlabel('Amount of available testing images')
    plt.ylabel('Number of classes')

    # Where we want the ticks, in pixel locations
    ticks = [int(el) for el in list(np.linspace(0, 200, 21))]
    # What those pixel locations correspond to in data coordinates.
    # Also set the float format here
    ax1.set_xticks(ticks)
    labels = ax1.get_xticklabels()
    plt.setp(labels, rotation=30)

    min_examples = 0
    ax1.hist(class_counts, bins=range(min_examples, max_data + 1, bin_size))
    # plt.show()
    filename = '{}.pdf'.format('data-dist')
    plt.savefig(filename)
    print("Plot has been saved as {}".format(filename))

    symbolid2latex = get_symbolid2latex()

    top10 = sorted(classes.items(), key=lambda n: n[1], reverse=True)[:10]
    top10_data = 0
    for index, count in top10:
        print("\t%s:\t%i" % (symbolid2latex[index2symbol_id[index]], count))
        top10_data += count
    total_data = sum([count for index, count in classes.items()])
    print("Top-10 has %i training data (%0.2f%% of total)" %
          (top10_data, float(top10_data) * 100.0 / total_data))
    print("%i classes have more than %i data items." %
          (sum([1 for _, count in classes.items() if count > max_data]),
           max_data))
    print(top10)


def _analyze_correlation(csv_filepath):
    """
    Analyze and visualize the correlation of features.
    Takes 1.5h.

    Parameters
    ----------
    csv_filepath : str
        Path to a CSV file which points to images
    """
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    symbol_id2index = generate_index(csv_filepath)
    data, y = load_images(csv_filepath,
                          symbol_id2index,
                          one_hot=False,
                          flatten=True)
    df = pd.DataFrame(data=data)

    print("Data loaded. Start correlation calculation. Takes 1.5h.")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Where we want the ticks, in pixel locations
    ticks = np.linspace(0, 1024, 17)
    # What those pixel locations correspond to in data coordinates.
    # Also set the float format here
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    labels = ax1.get_xticklabels()
    plt.setp(labels, rotation=30)

    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    # Add colorbar, make sure to specify tick locations to match desired
    # ticklabels
    fig.colorbar(cax, ticks=[-0.15, 0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1])
    filename = '{}.pdf'.format('feature-correlation')
    plt.savefig(filename)

def data_by_class(data):
    """
    Organize `data` by class.

    Parameters
    ----------
    data : list of dicts
        Each dict contains the key `symbol_id` which is the class label.

    Returns
    -------
    dbc : dict
        mapping class labels to lists of dicts
    """
    dbc = {}
    for item in data:
        if item['symbol_id'] in dbc:
            dbc[item['symbol_id']].append(item)
        else:
            dbc[item['symbol_id']] = [item]
    return dbc

def _is_valid_png(filepath):
    """
    Check if the PNG image is valid.

    Parameters
    ----------
    filepath : str
        Path to a PNG image

    Returns
    -------
    bool : True if the PNG image is valid, otherwise False.
    """
    try:
        test = Image.open(filepath)
        test.close()
        return True
    except:
        return False

def _verify_all():
    """Verify all PNG files in the training and test directories."""
    for csv_data_path in ['classification-task/fold-1/test.csv',
                          'classification-task/fold-1/train.csv']:
        train_data = load_csv(csv_data_path)
        for data_item in train_data:
            if not _is_valid_png(data_item['path']):
                logging.info("%s is invalid." % data_item['path'])
        print("Checked %i items of %s." %
                     (len(train_data), csv_data_path))
        

def create_stratified_split(csv_filepath='benat-data/train.csv', n_splits=5, kdirectory='benat-data/cv'):
    """
    Create a stratified split for the classification task.

    Parameters
    ----------
    csv_filepath : str
        Path to a CSV file which points to images
    n_splits : int
        Number of splits to make
    kdirectory : str
        Path to a directory where the output cvs-s will ve saved
    """
    from sklearn.model_selection import StratifiedKFold
    data = load_csv(csv_filepath)
    labels = [el['symbol_id'] for el in data]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    i = 1
    # kdirectory = 'classification-task'
    if not os.path.exists(kdirectory):
            os.makedirs(kdirectory)
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        # Note that providing labels (y) is sufficient to generate the splits and hence np.zeros(n_samples) may be used as a placeholder for X instead of actual training data.
        print("Create fold %i" % i)
        directory = "%s/fold-%i" % (kdirectory, i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print("Directory '%s' already exists. Please remove it." %
                  directory)
        i += 1
        train = [data[el] for el in train_index]
        test_ = [data[el] for el in test_index]
        for dataset, name in [(train, 'train'), (test_, 'test')]:
            with open(f"{directory}/{name}.csv", 'wt') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(('path', 'symbol_id', 'latex', 'user_id'))
                for el in dataset:
                    csv_writer.writerow((el['path'],
                                         el['symbol_id'],
                                         el['latex'],
                                         el['user_id']))



def create_stratified_train_test(csv_filepath, kdirectory='benat-data'):
    """
    Create a stratified split for the classification task.
    Modified version of create_stratified_split(), by Beñat Berasategui.

    Parameters
    ----------
    csv_filepath : str
        Path to a CSV file which points to images
    kdirectory : str
        Path to a directory where the output cvs-s will ve saved
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    data = load_csv(csv_filepath)
    labels = [el['symbol_id'] for el in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    i = 1
    # kdirectory = 'classification-task'
    if not os.path.exists(kdirectory):
            os.makedirs(kdirectory)
    for train_index, test_index in sss.split(np.zeros(len(labels)), labels): 
        # Note that providing labels (y) is sufficient to generate the splits and hence np.zeros(n_samples) may be used as a placeholder for X instead of actual training data.
        print("Create fold %i" % i)
        #directory = "%s/fold-%i" % (kdirectory, i)
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #else:
        #    print("Directory '%s' already exists. Please remove it." %
        #          directory)
        i += 1
        train = [data[el] for el in train_index]
        test_ = [data[el] for el in test_index]
        for dataset, name in [(train, 'train'), (test_, 'test')]:
            with open(kdirectory+f"/{name}.csv", 'wt') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(('path', 'symbol_id', 'latex', 'user_id'))
                for el in dataset:
                    csv_writer.writerow((el['path'],
                                         el['symbol_id'],
                                         el['latex'],
                                         el['user_id']))



