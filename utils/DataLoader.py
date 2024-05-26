import scipy.io

def loadData(data_name):
    """
    Load multi-view data with .mat
    """
    data = scipy.io.loadmat(data_name)
    features = data['X']
    gnd = data['Y']
    gnd = gnd.flatten()
    return features, gnd


def loadMGW(data_name, keyword):
    """
    Load multi-view graphs with .mat
    """
    data = scipy.io.loadmat(data_name)
    MGW = data[keyword]

    return MGW[0]

# def loadMGW(data_name):
#     """
#     Load multi-view graphs with .mat
#     """
#     data = scipy.io.loadmat(data_name)
#     MGW = data['HWPer01nanMGW']
#     return MGW

def loadMGW1(data_name):
    data = scipy.io.loadmat(data_name)
    MGW1 = data['G1']
    return MGW1

def loadMGW2(data_name):
    data = scipy.io.loadmat(data_name)
    MGW2 = data['G2']
    return MGW2

def loadMGW3(data_name):
    data = scipy.io.loadmat(data_name)
    MGW3 = data['G3']
    return MGW3

def loadMGW4(data_name):
    data = scipy.io.loadmat(data_name)
    MGW4 = data['G4']
    return MGW4

