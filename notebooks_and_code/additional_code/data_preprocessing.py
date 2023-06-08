import pandas as pd
import numpy as np
from Bio.UniProt.GOA import _gpa11iterator
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import random
import time
import gzip
import os
from os.path import join

CURRENT_DIR = os.getcwd() 
datasets_PubChem = join(CURRENT_DIR, ".." ,"additional_data_ESP", "substrate_synonyms")
mol_folder = join(CURRENT_DIR, ".." ,"additional_data_ESP", "mol-files")

df_UID_MID = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "enzyme_substrate_data", "df_UID_MID.pkl"))
df_chebi_to_inchi = pd.read_csv(join(CURRENT_DIR, ".." ,"data", "substrate_data", "chebiID_to_inchi.tsv"), sep = "\t")


def get_info_from_GO_Term(GO_Term):
    definition =  GO_Term[GO_Term.index("\ndef")+6:]
    definition = definition[:definition.index("\n")]
    
    name = GO_Term[GO_Term.index("\nname")+7:]
    name = name[:name.index("\n")]
    
    ID = GO_Term[GO_Term.index("\nid")+5:]
    ID = ID[:ID.index("\n")]
    return(definition, name, ID)

def get_RHEA_reaction_IDs(GO_ID, GO_Term):
    RHEA = np.nan
    if "xref: RHEA:" in GO_Term:
        RHEA = GO_Term[GO_Term.find("xref: RHEA:") + len("xref: RHEA:"):]
        RHEA = RHEA[:RHEA.find("\n")]
    return(RHEA)

digits = [str(i) + " " for i in range(1,100)] + ["a "]
def process_metabolites(metabolites_list):
    for i in range(len(metabolites_list)):
        met = metabolites_list[i]
        if met[0:2] in digits:
            met = met[2:]
        elif met[0:3] in digits:
            met = met[3:]
            
        if met[-1] == "(":      
            met = met[:-1]
        
        metabolites_list[i] = met
    return(metabolites_list)


#Definitions of GO Terms with information about enzyme catalyzed reaction contain one of the following
#After these starters the formula of the chemical reaction starts.
starter = ["OBSOLETE. Catalysis of the reaction: ", 
           "OBSOLETE. Catalysis of the reactions: ",
           "OBSOLETE. Catalysis of the reaction:", 
           "OBSOLETE. Catalysis of the reaction ",
           "Catalysis of the reaction: ", 
           "Catalysis of the reactions: ",
           "Catalysis of the reaction:", 
           "Catalysis of the reaction "]

def find_substrates(definition):
    substrates = []
    
    #trying to find one of the "starters" until successfull or until all startes have been tried:
    i = 0
    successfull = False
    while not successfull and i < len(starter):
        start = starter[i]
        try: 
            definition.index(start)
            definition = definition[len(start)+1:definition.index('."')]
            if "<=>" in definition:
                substrates = definition.split(" <=> ")[0]
            elif "->" in definition:
                substrates = definition.split(" -> ")[0]
            elif "=>" in definition:
                substrates = definition.split(" => ")[0]
            elif "= " in definition:
                substrates = definition.split("= ")[0]
            elif "=" in definition:
                substrates = definition.split("=")[0]
            else:
                print("Could not find substrate in the following definition: %s" % definition)
            substrates = substrates.replace(" + ", ";")
            substrates = substrates.split(";")
            substrates = progress_metabolites(metabolites_list = substrates)
            successfull = True
        except:
            pass
        i = i+1
        
    return(substrates)

def extract_RHEA_ID_and_CHEBI_IDs(entry):
    RHEA_ID = entry[0][len("ENTRY"): -1]
    RHEA_ID = RHEA_ID.split(" ")[-1]
    CHEBI_IDs = entry[2][len("EQUATION"): -1]
    CHEBI_IDs = CHEBI_IDs[CHEBI_IDs.index("CHEBI"):]
    return(RHEA_ID, CHEBI_IDs)

def get_substrate_IDs(IDs):
    IDs = IDs.split(" = ")[0]
    IDs = IDs.split(" => ")[0]
    IDs = IDs.split(" <=> ")[0]
    IDs = IDs.replace(" + ", ";")
    IDs = IDs.split(";")
    return([ID.split(" ")[-1] for ID in IDs])




def substrate_names_to_Pubchem_CIDs(metabolites):
    """
    A function that maps a list of metabolites to PubChem Compound IDs (CIDs), if there is an excat match
    for the metabolite and a synonym from the Pubchem synonym list.
    """    
    
    n = len(metabolites)
    match = [np.nan] * n

    for k in range(5):
        print("loading part %s of 5 of the synonym list..." %(k+1))
        df = pd.read_pickle(join(datasets_PubChem, "substrates_synonyms_part"+str(k)+".pkl"))
        substrates = list(df["substrates"])
        cid = list(df["CID"])
        df = None
        print("searching in synoynm list part %s for matches" %(k+1))
        
        for i in range(n):
            if pd.isnull(match[i]):
                met = metabolites[i].lower()
                if not pd.isnull(met):
                    try:
                        pos = substrates.index(met.lower())
                        match[i] = cid[pos]
                    except ValueError:
                        None
    df = pd.DataFrame(data= {"Metabolite" : metabolites, "CID" : match})
    return(df)


df = pd.read_pickle(join(CURRENT_DIR, ".." ,"data", "GOA_data", "df_GO_catalytic.pkl"))
catalytic_go_terms = list(set(df["GO ID"]))
ECO_to_GAF = pd.read_csv(join(CURRENT_DIR, ".." ,"data", "GOA_data", 'ECO_to_GAF.tsv'), sep = "\t")
exp_evidence = ["EXP","IDA","IPI","IMP","IGI","IEP", "HTP","HDA","HMP","HGI","HEP"]
phylo_evidence = ["IBA","IBD","IKR","IRD"]


def search_GOA_database(run):
    df_GO_UID = pd.DataFrame(columns = ["Uniprot ID", "GO Term", 'ECO_Evidence_code', 'evidence'])

    overall_count = 0
    filename = join(CURRENT_DIR, ".." ,"data", "GOA_data", 'goa_uniprot_all.gpa.gz')
    with gzip.open(filename, 'rt') as fp:
        for annotation in _gpa11iterator(fp):                 
            overall_count += 1
            if overall_count >= run*10**6 and overall_count < (run+1)*10**6:
                # Output annotated protein ID   
                UID = annotation['DB_Object_ID']
                GO_ID = annotation['GO_ID']
                ECO_Evidence_code = annotation["ECO_Evidence_code"]
                try:
                    evidence = list(ECO_to_GAF.loc[ECO_to_GAF["ECO"] == ECO_Evidence_code]["Evidence"])[0]
                except IndexError:
                    evidence = ""
                if GO_ID in catalytic_go_terms:   
                    if evidence in exp_evidence:
                        df_GO_UID = df_GO_UID.append({"Uniprot ID" : UID, "GO Term" : GO_ID,
                                                     'ECO_Evidence_code' : ECO_Evidence_code,
                                                      'evidence': "exp"}, ignore_index = True)

                    elif evidence in phylo_evidence:
                        df_GO_UID = df_GO_UID.append({"Uniprot ID" : UID, "GO Term" : GO_ID,
                                                     'ECO_Evidence_code' : ECO_Evidence_code,
                                                      'evidence': "phylo"}, ignore_index = True)

    df_GO_UID.to_pickle(join(CURRENT_DIR, ".." ,"data", "GOA_data", "experimental_and_phylogenetic",
                             "experimental_phylogenetic_df_GO_UID_part_" + str(run) +".pkl"))   




####Code for creating cluster of enzyme by enzyme sequence identity. Code was created by Martin Engqvist:
import os
import re
import argparse
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from Bio import SeqIO
from os.path import join, exists, abspath, isdir, dirname
import subprocess

def remove_header_gaps(folder, infile, outfile):
    '''
    CD-HIT truncates fasta record headers at whitespace,
    need to remove these before I run the algorithm
    '''
    if not exists(outfile):
        with open(join(folder, outfile), 'w') as f:
            for record in SeqIO.parse(join(folder, infile), 'fasta'):
                header = record.description.replace(' ', '_')
                seq = str(record.seq)

                f.write('>%s\n%s\n' % (header, seq))


def run_cd_hit(infile, outfile, cutoff, memory):
    '''
    Run a specific cd-hit command
    '''
    # get the right word size for the cutoff
    if cutoff < 0.5:
        word = 2
    elif cutoff < 0.6:
        word = 3
    elif cutoff < 0.7:
        word = 4
    else:
        word = 5

    mycmd = '%s -i %s -o %s -c %s -n %s -T 1 -M %s -d 0' % ('cd-hit', infile, outfile, cutoff, word, memory)
    print(mycmd)
    process = subprocess.Popen(mycmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    
def cluster_all_levels_60(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 1.0
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.9
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_100.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_90.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.7
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_80.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.6
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_70.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
        
def cluster_all_levels_80(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 1.0
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.9
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_100.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_90.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)



def cluster_all_levels(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 1.0
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.9
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_100.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_90.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.7
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_80.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.6
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_70.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.5
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_60.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.4
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(cluster_folder, '%s_clustered_sequences_50.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)


def parse_cd_hit(path_to_clstr):
    '''
    Gather the clusters of CD-hit output `path_to_clust` into a dict.
    '''
    # setup regular expressions for parsing
    pat_id = re.compile(r">(.+?)\.\.\.")
    is_center = re.compile(r">(.+?)\.\.\. \*")

    with open(path_to_clstr) as f:
        clusters = {}
        cluster = []
        id_clust = None
        next(f)  # advance first cluster header
        for line in f:
            if line.startswith(">"):
                # if cluster ended, flush seq ids to it
                clusters[id_clust] = cluster
                cluster = []
                continue
            match = pat_id.search(line)
            if match:
                if is_center.search(line):
                    id_clust = match[1]
                else:
                    cluster.append(match[1])
        clusters[id_clust] = cluster
    return clusters


def scale_up_cd_hit(paths_to_clstr):
    '''
    Hierarchically expand CD-hit clusters.

    Parameters
    ----------
    paths_to_clstr: list[str]
        paths to rest of the cd-hit output files, sorted by
        decreasing similarity (first is 100).

    Output
    ------
    clust_now: dict
        id: ids

    '''
    clust_above = parse_cd_hit(paths_to_clstr[0])

    for path in paths_to_clstr[1:]:
        clust_now = parse_cd_hit(path)
        for center in clust_now:
            clust_now[center] += [
                seq
                for a_center in clust_now[center] + [center]
                for seq in clust_above[a_center]
            ]
        clust_above = clust_now

    return clust_above


def find_cluster_members(folder, filename):
    '''
    Go through the cluster files and collect
    all the cluster members, while indicating
    which belongs where.
    '''
    # get a list of filenames
    CLUSTER_FILES = [
        join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
        for sim in [100, 90, 80, 70, 60, 50, 40]
    ]

    # collect all cluster members
    clusters = scale_up_cd_hit(CLUSTER_FILES)
    ind_clusters = {}
    i = 0
    for clus in clusters:
        ind_clusters[i] = [clus] + clusters[clus]
        i += 1

    # convert to format that is suitable for data frames
    clusters_for_df = {'cluster': [], 'member': []}
    for ind in ind_clusters:
        for member in ind_clusters[ind]:
            clusters_for_df['cluster'].append(ind)
            clusters_for_df['member'].append(member)

    df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

    return df


def find_cluster_members_80(folder, filename):
    '''
    Go through the cluster files and collect
    all the cluster members, while indicating
    which belongs where.
    '''
    # get a list of filenames
    CLUSTER_FILES = [
        join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
        for sim in [100, 90, 80]
    ]

    # collect all cluster members
    clusters = scale_up_cd_hit(CLUSTER_FILES)
    ind_clusters = {}
    i = 0
    for clus in clusters:
        ind_clusters[i] = [clus] + clusters[clus]
        i += 1

    # convert to format that is suitable for data frames
    clusters_for_df = {'cluster': [], 'member': []}
    for ind in ind_clusters:
        for member in ind_clusters[ind]:
            clusters_for_df['cluster'].append(ind)
            clusters_for_df['member'].append(member)

    df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

    return df

def find_cluster_members_60(folder, filename):
    '''
    Go through the cluster files and collect
    all the cluster members, while indicating
    which belongs where.
    '''
    # get a list of filenames
    CLUSTER_FILES = [
        join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
        for sim in [100, 90, 80,70,60]
    ]

    # collect all cluster members
    clusters = scale_up_cd_hit(CLUSTER_FILES)
    ind_clusters = {}
    i = 0
    for clus in clusters:
        ind_clusters[i] = [clus] + clusters[clus]
        i += 1

    # convert to format that is suitable for data frames
    clusters_for_df = {'cluster': [], 'member': []}
    for ind in ind_clusters:
        for member in ind_clusters[ind]:
            clusters_for_df['cluster'].append(ind)
            clusters_for_df['member'].append(member)

    df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

    return df


def kfold_by(df, key, k=5):
    """K-Split dataset `df` by values in `key` into `k` groups.

    Parameters
    ----------
    df: pandas.DataFrame
    key: str
        columns to use as splitting
    k: int
        number of groups.

    Returns
    -------
    k*(groups): pandas.DataFrame
        each df is the training set of the fold

    """
    kf = KFold(n_splits=k, random_state=4321, shuffle=True)
    set_keys = np.unique(df[key])
    return [
        df[df[key].isin(set_keys[train_index])]
        for train_index, _ in kf.split(set_keys)
    ]


def split_by(df, key, frac=0.8):
    """Split dataset `df` by values in `key`.

    Parameters
    ----------
    df: pandas.DataFrame
    key: str
        columns to use as splitting
    frac: float
        fraction of `key` groups into `df`.

    Returns
    -------
    (train, test, valid): pandas.DataFrames

    """
    # shuffle the data frame
    df = df.sample(frac=1, random_state=4321).reset_index(drop=True)

    # get all the unique identifiers
    set_keys = np.unique(df[key])

    # get the training identifiers
    train_clusters = np.random.choice(
        set_keys, size=int(len(set_keys) * frac), replace=False
    )
    train = df[df[key].isin(train_clusters)]

    # from the remaining ones, put half as validation and half as test
    remaining = df[~df.index.isin(train.index)]
    # valid and test sets will have equal sizes of 1-frac
    # at this point we are not worried about `key` anymore
    valid = remaining.sample(frac=1 / 2)
    test = remaining[~remaining.index.isin(valid.index)]
    return train, test, valid


def make_splits(folder, df):
    '''
    Takes an input data frame with information on cluster
    belongings and generates train/validation/test splits for DL.
    '''
    # make train/validation/test splits for DL
    train, validation, test = split_by(df, "cluster", frac=0.8)

    train.drop('cluster', axis=1).to_csv(join(folder, f"split_training.tsv"),
                 sep="\t", index=False, header=False)

    validation.drop('cluster', axis=1).to_csv(join(folder, f"split_validation.tsv"),
                      sep="\t", index=False, header=False)

    test.drop('cluster', axis=1).to_csv(join(folder, "split_test.tsv"),
                sep="\t", index=False, header=False)





def get_mol(met_ID):
    is_CHEBI_ID = (met_ID[0:5] == "CHEBI")
    is_InChI = (met_ID[0:5] == "InChI")
    if is_CHEBI_ID:
        try:
            ID = int(met_ID.split(" ")[0].split(":")[-1])
            Inchi = list(df_chebi_to_inchi["Inchi"].loc[df_chebi_to_inchi["ChEBI"] == float(ID)])[0]
            mol = Chem.inchi.MolFromInchi(Inchi)
        except:
            mol = None     
    elif is_InChI:
        try:
            mol = Chem.inchi.MolFromInchi(met_ID)
        except:
            mol = None
        
    else:
        try:
            mol = Chem.MolFromMolFile(mol_folder +  "mol-files\\" + met_ID + '.mol')
        except OSError:
            mol = None
            
    return(mol)

def drop_samples_without_mol_file(df):
    droplist = []
    for ind in df.index:
        if get_mol(met_ID = df["molecule ID"][ind]) is None:
            droplist.append(ind)

    df.drop(droplist, inplace = True)
    return(df)

def get_metabolites_and_similarities(df):
    df_metabolites = pd.DataFrame(data = {"ECFP": df["ECFP"], "ID": df["molecule ID"]})
    df_metabolites = df_metabolites.drop_duplicates()
    df_metabolites.reset_index(inplace = True, drop = True)


    ms = [get_mol(met_ID = df_metabolites["ID"][ind]) for ind in df_metabolites.index]
    fps = [Chem.RDKFingerprint(x) for x in ms]

    similarity_matrix = np.zeros((len(ms), len(ms)))
    for i in range(len(ms)):
        for j in range(len(ms)):
            similarity_matrix[i,j] = DataStructs.FingerprintSimilarity(fps[i],fps[j])
            
    return(df_metabolites, similarity_matrix)



def get_valid_list(met_ID, UID, forbidden_metabolites, df_metabolites, similarity_matrix, lower_bound =0.7, upper_bound =0.9):
    binding_met_IDs = list(df_UID_MID["molecule ID"].loc[df_UID_MID["Uniprot ID"] == UID])
    k = df_metabolites.loc[df_metabolites["ID"] == met_ID].index[0]

    similarities = similarity_matrix[k,:]
    selection = (similarities< upper_bound) * (similarities >lower_bound) 
    metabolites = list(df_metabolites["ID"].loc[selection])
    
    no_mets = list(set(binding_met_IDs + forbidden_metabolites))
    
    metabolites = [met for met in metabolites if (met not in no_mets)]
    return(metabolites)


def create_negative_samples(df, df_metabolites, similarity_matrix):
    start = time.time()
    UID_list = []
    MID_list = []
    Type_list = []
    forbidden_mets = []

    for ind in df.index:
        if ind % 100 ==0:
            print(ind)
            print("Time: %s [min]" % np.round(float((time.time()-start)/60),2))

            df2 = pd.DataFrame(data = {"Uniprot ID": UID_list, "molecule ID" : MID_list, "type" : Type_list})
            df2["Binding"] = 0
            df = pd.concat([df, df2], ignore_index=True)

            UID_list, MID_list, Type_list = [], [], []

            forbidden_mets_old = forbidden_mets.copy()
            all_mets = list(set(df["molecule ID"]))
            all_mets = [met for met in all_mets if not met in forbidden_mets_old]
            forbidden_mets = list(set([met for met in all_mets if 
                                       (np.mean(df["Binding"].loc[df["molecule ID"] == met]) < 1/4)]))
            forbidden_mets = forbidden_mets + forbidden_mets_old
            print(len(forbidden_mets))

        UID = df["Uniprot ID"][ind]
        Type = df["type"][ind]
        met_ID = df["molecule ID"][ind]

        metabolites = get_valid_list(met_ID = met_ID, UID = UID, forbidden_metabolites= forbidden_mets,
                                     df_metabolites = df_metabolites, similarity_matrix = similarity_matrix,
                                     lower_bound =0.7, upper_bound =0.95)
        lower_bound = 0.7
        while len(metabolites) < 2:
            lower_bound = lower_bound - 0.2
            metabolites = get_valid_list(met_ID = met_ID, UID = UID, forbidden_metabolites= forbidden_mets,
                                     df_metabolites = df_metabolites, similarity_matrix = similarity_matrix,
                                     lower_bound =lower_bound, upper_bound =0.95)
            if lower_bound <0:
                break
        
        new_metabolites =  random.sample(metabolites, min(3,len(metabolites)))

        for met in new_metabolites:
            UID_list.append(UID), MID_list.append(met), Type_list.append(Type)

    df2 = pd.DataFrame(data = {"Uniprot ID": UID_list, "molecule ID" : MID_list, "type" : Type_list})
    df2["Binding"] = 0

    df = pd.concat([df, df2], ignore_index = True)
    return(df)