"""
Calculate power transfer distribution factors
"""

import numpy as np
import pandas as pd
    
def calc_ptdf(lines:object, ref_node:int=0):
    """
    input: lines - pandas dataframe
        	ref_node - referance bus
    """
    no_nodes = max([lines['from'].max(),lines['to'].max()])+1    
    no_lines = lines.shape[0]
    nodes = list(range(no_nodes))
    nodes.remove(ref_node)
    # find connections:
    A = np.array([
        [1 if i==lines['from'][r] else (-1 if i==lines['to'][r] else 0) for i in range(no_nodes)]
    for r in range(no_lines)])
    # remove reference bus:
    A2 = A.T[nodes].T
    # find susceptances:
    B = 1/lines['reactance'].to_numpy()
    B = B.reshape(-1,1)
    # calculate ptdfs:
    PTDF_nohub = np.dot((B*A2),np.linalg.inv(np.dot(A2.T,B*A2)))
    # add reference bus:
    PTDF = np.insert(PTDF_nohub, ref_node, np.zeros(no_lines), 1)
    return PTDF
    
if __name__ == "__main__":
    lines = pd.DataFrame(
        [
            [np.random.random()/10, 0.5, np.random.random()*100, 0, 2],
            [np.random.random()/10, 0.1, np.random.random()*100, 0, 1],
            [np.random.random()/10, 0.2, np.random.random()*100, 1, 2]
        ], columns = ['resistance', 'reactance', 'capacity', 'from', 'to'])
    
    calc_ptdf(lines)
