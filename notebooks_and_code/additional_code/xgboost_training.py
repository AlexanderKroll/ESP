from rdkit import Chem
import pandas as pd
import pickle
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CURRENT_DIR = os.getcwd()


df_chebi_to_inchi = pd.read_csv(join(CURRENT_DIR, ".." ,"data", "substrate_data", "chebiID_to_inchi.tsv"), sep = "\t")
mol_folder = "C:\\Users\\alexk\\mol-files\\"


def create_target_dict(df, target_variable_dict):
    for ind in df.index:
        uid = df["Uniprot ID"][ind]
        cid = df["molecule ID"][ind]
        target_variable_dict[uid + "_" + cid.replace(":", "_")] = df["Binding"][ind]
    return(target_variable_dict)

def get_uid_cid_IDs(df):
    ID_list = []
    for ind in df.index:
        uid = df["Uniprot ID"][ind]
        cid = df["molecule ID"][ind]
        ID_list.append(uid + "_" + cid)
    return(ID_list)

def calculate_atom_and_bond_feature_vectors(mol_files):
    #check if feature vectors have already been calculated:
    try:
        os.mkdir(join(CURRENT_DIR, ".." ,"data", "substrate_data", "mol_feature_vectors"))
    except FileExistsError:
        None
    
    #existing feature vector files:
    feature_files = os.listdir(join(CURRENT_DIR, ".." ,"data", "substrate_data", "mol_feature_vectors"))
    for mol_file in mol_files:
        #check if feature vectors were already calculated:
        if not mol_file + "-atoms.txt" in  feature_files:
            #load mol_file
            is_CHEBI_ID = (mol_file[0:5] == "CHEBI")
            if is_CHEBI_ID:
                ID = int(mol_file.split(" ")[0].split(":")[-1])
                Inchi = list(df_chebi_to_inchi["Inchi"].loc[df_chebi_to_inchi["ChEBI"] == float(ID)])[0]
                
                if not pd.isnull(Inchi):
                    mol = Chem.inchi.MolFromInchi(Inchi)
                else:
                    print(ID, Inchi)
            else:
                mol = Chem.MolFromMolFile(mol_folder +  "/mol-files/" + mol_file + '.mol')
            if not mol is None:
                calculate_atom_feature_vector_for_mol_file(mol, mol_file)
                calculate_bond_feature_vector_for_mol_file(mol, mol_file)
                
def calculate_atom_feature_vector_for_mol_file(mol, mol_file):
    #get number of atoms N
    N = mol.GetNumAtoms()
    atom_list = []
    for i in range(N):
        features = []
        atom = mol.GetAtomWithIdx(i)
        features.append(atom.GetAtomicNum()), features.append(atom.GetDegree()), features.append(atom.GetFormalCharge())
        features.append(str(atom.GetHybridization())), features.append(atom.GetIsAromatic()), features.append(atom.GetMass())
        features.append(atom.GetTotalNumHs()), features.append(str(atom.GetChiralTag()))
        atom_list.append(features)
    with open(join(CURRENT_DIR, ".." ,"data", "substrate_data", "mol_feature_vectors", mol_file.replace(":", "_") + "-atoms.txt"), "wb") as fp:   #Pickling
        pickle.dump(atom_list, fp)
            
def calculate_bond_feature_vector_for_mol_file(mol, mol_file):
    N = mol.GetNumBonds()
    bond_list = []
    for i in range(N):
        features = []
        bond = mol.GetBondWithIdx(i)
        features.append(bond.GetBeginAtomIdx()), features.append(bond.GetEndAtomIdx()),
        features.append(str(bond.GetBondType())), features.append(bond.GetIsAromatic()),
        features.append(bond.IsInRing()), features.append(str(bond.GetStereo()))
        bond_list.append(features)
    with open(join(CURRENT_DIR, ".." ,"data", "substrate_data", "mol_feature_vectors", mol_file.replace(":", "_") + "-bonds.txt"), "wb") as fp:   #Pickling
        pickle.dump(bond_list, fp)


N = 70 #maximal number of atoms in a molecule
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1 + F2

try:
    os.mkdir(join(CURRENT_DIR, ".." ,"data", "substrate_data","GNN_input_matrices"))
except FileExistsError:
    None

    
#Create dictionaries for the bond features:
dic_bond_type = {'AROMATIC': np.array([0,0,0,1]), 'DOUBLE': np.array([0,0,1,0]),
                 'SINGLE': np.array([0,1,0,0]), 'TRIPLE': np.array([1,0,0,0])}

dic_conjugated =  {0.0: np.array([0]), 1.0: np.array([1])}

dic_inRing = {0.0: np.array([0]), 1.0: np.array([1])}

dic_stereo = {'STEREOANY': np.array([0,0,0,1]), 'STEREOE': np.array([0,0,1,0]),
              'STEREONONE': np.array([0,1,0,0]), 'STEREOZ': np.array([1,0,0,0])}


##Create dictionaries, so the atom features can be easiliy converted into a numpy array

#all the atomic numbers with a total count of over 200 in the data set are getting their own one-hot-encoded
#vector. All the otheres are lumped to a single vector.
dic_atomic_number = {0.0: np.array([1,0,0,0,0,0,0,0,0,0]), 1.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     3.0: np.array([0,0,0,0,0,0,0,0,0,1]),  4.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     5.0: np.array([0,0,0,0,0,0,0,0,0,1]),  6.0: np.array([0,1,0,0,0,0,0,0,0,0]),
                     7.0:np.array([0,0,1,0,0,0,0,0,0,0]),  8.0: np.array([0,0,0,1,0,0,0,0,0,0]),
                     9.0: np.array([0,0,0,0,1,0,0,0,0,0]), 11.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     12.0: np.array([0,0,0,0,0,0,0,0,0,1]), 13.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     14.0: np.array([0,0,0,0,0,0,0,0,0,1]), 15.0: np.array([0,0,0,0,0,1,0,0,0,0]),
                     16.0: np.array([0,0,0,0,0,0,1,0,0,0]), 17.0: np.array([0,0,0,0,0,0,0,1,0,0]),
                     19.0: np.array([0,0,0,0,0,0,0,0,0,1]), 20.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     23.0: np.array([0,0,0,0,0,0,0,0,0,1]), 24.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     25.0: np.array([0,0,0,0,0,0,0,0,0,1]), 26.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     27.0: np.array([0,0,0,0,0,0,0,0,0,1]), 28.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     29.0: np.array([0,0,0,0,0,0,0,0,0,1]), 30.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     32.0: np.array([0,0,0,0,0,0,0,0,0,1]), 33.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     34.0: np.array([0,0,0,0,0,0,0,0,0,1]), 35.0: np.array([0,0,0,0,0,0,0,0,1,0]),
                     37.0: np.array([0,0,0,0,0,0,0,0,0,1]), 38.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     42.0: np.array([0,0,0,0,0,0,0,0,0,1]), 46.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     47.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     48.0: np.array([0,0,0,0,0,0,0,0,0,1]), 50.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     51.0: np.array([0,0,0,0,0,0,0,0,0,1]), 52.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     53.0: np.array([0,0,0,0,0,0,0,0,0,1]), 54.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     56.0: np.array([0,0,0,0,0,0,0,0,0,1]), 57.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     74.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     78.0: np.array([0,0,0,0,0,0,0,0,0,1]), 79.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     80.0: np.array([0,0,0,0,0,0,0,0,0,1]), 81.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     82.0: np.array([0,0,0,0,0,0,0,0,0,1]), 83.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     86.0: np.array([0,0,0,0,0,0,0,0,0,1]), 88.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     90.0: np.array([0,0,0,0,0,0,0,0,0,1]), 94.0: np.array([0,0,0,0,0,0,0,0,0,1])}

#There are only 5 atoms in the whole data set with 6 bonds and no atoms with 5 bonds. Therefore I lump 4, 5 and 6 bonds
#together
dic_num_bonds = {0.0: np.array([0,0,0,0,1]), 1.0: np.array([0,0,0,1,0]),
                 2.0: np.array([0,0,1,0,0]), 3.0: np.array([0,1,0,0,0]),
                 4.0: np.array([1,0,0,0,0]), 5.0: np.array([1,0,0,0,0]),
                 6.0: np.array([1,0,0,0,0])}

#Almost alle charges are -1,0 or 1. Therefore I use only positiv, negative and neutral as features:
dic_charge = {-4.0: np.array([1,0,0]), -3.0: np.array([1,0,0]),  -2.0: np.array([1,0,0]), -1.0: np.array([1,0,0]),
               0.0: np.array([0,1,0]),  1.0: np.array([0,0,1]),  2.0: np.array([0,0,1]),  3.0: np.array([0,0,1]),
               4.0: np.array([0,0,1]), 5.0: np.array([0,0,1]), 6.0: np.array([0,0,1])}

dic_hybrid = {'S': np.array([0,0,0,0,1]), 'SP': np.array([0,0,0,1,0]), 'SP2': np.array([0,0,1,0,0]),
              'SP3': np.array([0,1,0,0,0]), 'SP3D': np.array([1,0,0,0,0]), 'SP3D2': np.array([1,0,0,0,0]),
              'UNSPECIFIED': np.array([1,0,0,0,0])}

dic_aromatic = {0.0: np.array([0]), 1.0: np.array([1])}

dic_H_bonds = {0.0: np.array([0,0,0,1]), 1.0: np.array([0,0,1,0]), 2.0: np.array([0,1,0,0]),
               3.0: np.array([1,0,0,0]), 4.0: np.array([1,0,0,0]), 5.0: np.array([1,0,0,0]),
               6.0: np.array([1,0,0,0])}

dic_chirality = {'CHI_TETRAHEDRAL_CCW': np.array([1,0,0]), 'CHI_TETRAHEDRAL_CW': np.array([0,1,0]),
                 'CHI_UNSPECIFIED': np.array([0,0,1])}

def create_bond_feature_matrix(mol_name, N =70):
    '''create adjacency matrix A and bond feature matrix/tensor E'''
    try:
        with open(join(CURRENT_DIR, ".." ,"data", "substrate_data", "mol_feature_vectors",
                       mol_name + "-bonds.txt"), "rb") as fp:   # Unpickling
            bond_features = pickle.load(fp)
    except FileNotFoundError:
        return(None)
    A = np.zeros((N,N))
    E = np.zeros((N,N,10))
    for i in range(len(bond_features)):
        line = bond_features[i]
        start, end = line[0], line[1]
        A[start, end] = 1 
        A[end, start] = 1
        e_vw = np.concatenate((dic_bond_type[line[2]], dic_conjugated[line[3]],
                               dic_inRing[line[4]], dic_stereo[line[5]]))
        E[start, end, :] = e_vw
        E[end, start, :] = e_vw
    return(A,E)


def create_atom_feature_matrix(mol_name, N =70):
    try:
        with open(join(CURRENT_DIR, ".." ,"data", "substrate_data", "mol_feature_vectors",
                       mol_name + "-atoms.txt"), "rb") as fp:   # Unpickling
            atom_features = pickle.load(fp)
    except FileNotFoundError:
        print("File not found for %s" % mol_name)
        return(None)
    X = np.zeros((N,32))
    if len(atom_features) >=N:
        print("More than %s (%s) atoms in molcuele %s" % (N,len(atom_features), mol_name))
        return(None)
    for i in range(len(atom_features)):
        line = atom_features[i]
        try:
            atomic_number_mapping = dic_atomic_number[line[0]]
        except KeyError:
            atomic_number_mapping = np.array([0,0,0,0,0,0,0,0,0,1])
        x_v = np.concatenate((atomic_number_mapping, dic_num_bonds[line[1]], dic_charge[line[2]],
                             dic_hybrid[line[3]], dic_aromatic[line[4]], np.array([line[5]/100.]),
                             dic_H_bonds[line[6]], dic_chirality[line[7]]))
        X[i,:] = x_v
    return(X)


def concatenate_X_and_E(X, E, N = 70, F = 32+10):
    XE = np.zeros((N, N, F))
    for v in range(N):
        x_v = X[v,:]
        for w in range(N):
            XE[v,w, :] = np.concatenate((x_v, E[v,w,:]))
    return(XE)



def create_input_data_for_GNN_for_substrates(substrate_ID, print_error = False):
    try:
        x = create_atom_feature_matrix(mol_name = substrate_ID, N =N)
        if not x is None: 
            a,e = create_bond_feature_matrix(mol_name = substrate_ID, N =N)
            a = np.reshape(a, (N,N,1))
            xe = concatenate_X_and_E(x, e)
            return([np.array(xe), np.array(x), np.array(a)])
        else:
            if print_error:
                print("Could not create input for substrate ID %s" %substrate_ID)      
            return(None, None, None)
    except:
        print("Error for substrate ID %s" % substrate_ID)
        return(None, None, None)
    
    
def create_input_data_for_GNN_for_substrates(substrate_ID, print_error = False):
    try:
        x = create_atom_feature_matrix(mol_name = substrate_ID, N =N)
        if not x is None: 
            a,e = create_bond_feature_matrix(mol_name = substrate_ID, N =N)
            a = np.reshape(a, (N,N,1))
            xe = concatenate_X_and_E(x, e, N = 70)
            return([np.array(xe), np.array(x), np.array(a)])
        else:
            if print_error:
                print("Could not create input for substrate ID %s" %substrate_ID)      
            return(None, None, None)
    except:
        return(None, None, None)

    
    
def calculate_and_save_input_matrixes(molecule_ID, save_folder = join(CURRENT_DIR, ".." ,"data", "substrate_data",
                                                                      "GNN_input_matrices")):
    molecule_ID = molecule_ID.replace(":", "_")
    [XE, X, A] = create_input_data_for_GNN_for_substrates(substrate_ID = molecule_ID, print_error=True)
    if not A is None:
        np.save(join(save_folder, molecule_ID + '_X.npy'), X) #feature matrix of atoms/nodes
        np.save(join(save_folder, molecule_ID + '_XE.npy'), XE) #feature matrix of atoms/nodes and bonds/edges
        np.save(join(save_folder, molecule_ID + '_A.npy'), A) #adjacency matrix


# Model parameters
N = 70        # maximum number of nodes
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1+F2
D = 50
droprate = 0.2



class GNN(nn.Module):
    def __init__(self, D= 50, N = 70, F1 = 32 , F2 = 10, F = F1+F2, droprate = 0.2):
        super(GNN, self).__init__()
        #first head
        self.Wi = nn.Parameter(torch.empty((1,1,F,D), requires_grad = True).to(device))
        self.Wm1 = nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wm2= nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wa = nn.Parameter(torch.empty((1,D+F1,D), requires_grad = True).to(device))
        nn.init.normal_(self.Wa), nn.init.normal_(self.Wm1), nn.init.normal_(self.Wm2), nn.init.normal_(self.Wi)

        
        self.OnesN_N = torch.tensor(np.ones((N,N)), dtype = torch.float32, requires_grad = False).to(device)
        self.Ones1_N = torch.tensor(np.ones((1,N)), dtype = torch.float32, requires_grad = False).to(device)
        self.BN1 = nn.BatchNorm2d(D).to(device)
        self.BN2 = nn.BatchNorm2d(D).to(device)

        
        self.D = D
        #seconda head
        #self.BN2_esm1b = nn.BatchNorm1d(64).to(device)
        
        self.BN3 = nn.BatchNorm1d(D+50).to(device)
        self.linear1 = nn.Linear(D+50, 32).to(device)
        self.linear2 = nn.Linear(32, 1).to(device)
        
        #dropout_layer
        self.drop_layer = nn.Dropout(p= droprate)

    def forward(self, XE, X, A, ESM1b):
        X = X.view((-1, N, 1, F1))
        H0 = nn.ReLU()(torch.matmul(XE, self.Wi)) #W*XE
        #only get neighbors in each row: (elementwise multiplication)
        M1 = torch.mul(H0, A)
        M1 = torch.transpose(M1, dim0 =1, dim1 =2)
        M1 = torch.matmul(self.OnesN_N, M1)
        M1 = torch.add(M1, -torch.transpose(H0, dim0 =1, dim1 =2) )
        M1 = torch.mul(M1, A)
        H1 = torch.add(H0, torch.matmul(M1, self.Wm1))
        H1 = torch.transpose(H1, dim0 =1, dim1 =3)
        H1 = nn.ReLU()(self.BN1(H1))
        H1 = torch.transpose(H1, dim0 =1, dim1 =3)


        M2 = torch.mul(H1, A)
        M2 = torch.transpose(M2, dim0 =1, dim1 =2)
        M2 = torch.matmul(self.OnesN_N, M2)
        M2 = torch.add(M2, -torch.transpose(H1, dim0 =1, dim1 =2))
        M2 = torch.mul(M2, A)
        H2 = torch.add(H0, torch.matmul(M2, self.Wm2)) 
        H2 = torch.transpose(H2, dim0 =1, dim1 =3)
        H2 = nn.ReLU()(self.BN2(H2))
        H2 = torch.transpose(H2, dim0 =1, dim1 =3) 

        M_v = torch.mul(H2, A)
        M_v = torch.matmul(self.Ones1_N, M_v)
        XM = torch.cat((X, M_v),3)
        H = nn.ReLU()(torch.matmul(XM, self.Wa))
        h = torch.matmul(self.Ones1_N, torch.transpose(H, dim0 =1, dim1 =2))
        h = self.drop_layer(h.view((-1,self.D)))
        
        
        
        h = torch.cat((h, ESM1b),1)
        h =  nn.ReLU()(self.linear1(self.BN3(h)))
        y =nn.Sigmoid()(self.linear2(h))
        return(y)
    
    def get_GNN_rep(self, XE, X, A):
        X = X.view((-1, N, 1, F1))
        H0 = nn.ReLU()(torch.matmul(XE, self.Wi)) #W*XE
        #only get neighbors in each row: (elementwise multiplication)
        M1 = torch.mul(H0, A)
        M1 = torch.transpose(M1, dim0 =1, dim1 =2)
        M1 = torch.matmul(self.OnesN_N, M1)
        M1 = torch.add(M1, -torch.transpose(H0, dim0 =1, dim1 =2) )
        M1 = torch.mul(M1, A)
        H1 = torch.add(H0, torch.matmul(M1, self.Wm1))
        H1 = torch.transpose(H1, dim0 =1, dim1 =3)
        H1 = nn.ReLU()(self.BN1(H1))
        H1 = torch.transpose(H1, dim0 =1, dim1 =3) 


        M2 = torch.mul(H1, A)
        M2 = torch.transpose(M2, dim0 =1, dim1 =2)
        M2 = torch.matmul(self.OnesN_N, M2)
        M2 = torch.add(M2, -torch.transpose(H1, dim0 =1, dim1 =2))
        M2 = torch.mul(M2, A)
        H2 = torch.add(H0, torch.matmul(M2, self.Wm2)) 
        H2 = torch.transpose(H2, dim0 =1, dim1 =3)
        H2 = nn.ReLU()(self.BN2(H2))
        H2 = torch.transpose(H2, dim0 =1, dim1 =3) 

        M_v = torch.mul(H2, A)
        M_v = torch.matmul(self.Ones1_N, M_v)
        XM = torch.cat((X, M_v),3)
        H = nn.ReLU()(torch.matmul(XM, self.Wa))
        h = torch.matmul(self.Ones1_N, torch.transpose(H, dim0 =1, dim1 =2))
        h = h.view((-1,self.D))
        return(h)