import pickle


def save(network, path):
    with open(path, 'wb') as handle:
        pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(path):
    with open(path, 'rb') as handle:
        network = pickle.load(handle)
    return network