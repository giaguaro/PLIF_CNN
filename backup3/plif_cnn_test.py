
    
#!/usr/bin/env python
# coding: utf-8


import os
import sys
import re
import torch
from glob import glob
from openbabel import pybel
import numpy as np
import random
import pickle
import warnings
import requests
import os
import glob
import pandas as pd
import openbabel
import numpy as np
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split
from dask.dataframe import from_pandas
from dask.multiprocessing import get
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=250,use_memory_fs=False)


PYTORCH_ENABLE_MPS_FALLBACK=1
PYTORCH_ENABLE_SparseCPU_FALLBACK=1
warnings.filterwarnings("ignore")

from dask_jobqueue import SLURMCluster

train_grids=None
test_grids=None
rotations=9
full_batch=10
features_shape=94
# First training grids: 
train_label=[]
# Second testing grids: 
test_label=[]
    
# cluster = SLURMCluster(cores=1024,
#                        processes=12,
#                        memory="250GB",
#                       # account="hmslati",
#                       # walltime="01:00:00",
#                        queue="gpu-bigmem")

# PLEASE READ -> 45次实验分别进行10倍交叉验证，取平均

#Converts the protein-ligand complexes into 4D tensor. 
class Feature_extractor():
    def __init__(self):
        self.atom_codes = {}
        #'others' includs metal atoms and B atom. There are no B atoms on training and test sets. 
        # 89 to 102 will be reserved to PLIF features as follows:
        # 89: hydrophobic
        # 90: hbond
        # 91: waterbridge
        # 92: saltbridge
        # 93: pistacking
        # 94: pication
        # 95: halogen
        # 96: metal
        # 97: Distances 
        
        # rarely occuring elements: ['4', '13', '22-23', '25', '28', '32', '40', '45', '47-52', '55', '75-76', '78', '80', '82'] 
        others = ([4,13,25,28,32,40,45,55,78,80,82]+list(range(22,24))+list(range(47,53))+list(range(75,77)))
        plif_specs=list(range(89,98))
        #C and N atoms can be hybridized in three ways and S atom can be hybridized in two ways here. 
        #Hydrogen atom is also considered for feature extraction. I think phosphor atom has 3 or 5 as hyb states but 
        # in biological system its usually the same recurrent phosphate even in most small molecules so safe to assume one
        # hybridization state for this purpose. 
        atom_types = [1,(6,1),(6,2),(6,3),(7,1),(7,2),(7,3),8,15,(16,2),(16,3),
                      34,9,17,35,53,11,12,13,14,5,19,20,25,29,28,30,33,3,27,24,26,31,42,79,44,74,others]+plif_specs
      
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
                
            else:
                self.atom_codes[j] = i              
        
        self.sum_atom_types = len(atom_types)
        
    #Onehot encoding of each atom. The atoms in protein or ligand are treated separately.
    def encode(self, atomic_num, orig_coords, plifs, molprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if molprotein == 1:
            encoding[self.atom_codes[atomic_num]] = 1.0
            for coord, plif_feats in plifs.items():
                if [round(item) for item in coord] == [round(item) for item in orig_coords]:
                    encoding[self.atom_codes[89]] = 1.0 if plifs[coord][0] == 'hydrophobic' \
                    else 0.0
                    encoding[self.atom_codes[90]] = 1.0 if plifs[coord][0] == 'hbond' \
                    else 0.0
                    encoding[self.atom_codes[91]] = 1.0 if plifs[coord][0] == 'waterbridge' \
                    else 0.0
                    encoding[self.atom_codes[92]] = 1.0 if plifs[coord][0] == 'saltbridge' \
                    else 0.0
                    encoding[self.atom_codes[93]] = 1.0 if plifs[coord][0] == 'pistacking' \
                    else 0.0
                    encoding[self.atom_codes[94]] = 1.0 if plifs[coord][0] == 'pication' \
                    else 0.0
                    encoding[self.atom_codes[95]] = 1.0 if plifs[coord][0] == 'halogen' \
                    else 0.0
                    encoding[self.atom_codes[96]] = 1.0 if plifs[coord][0] == 'metal' \
                    else 0.0
                    
                    #distance
                    encoding[self.atom_codes[97]] = plifs[coord][1]
                    

                
        else:
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
            for coord, plif_feats in plifs.items():
                if [round(item) for item in coord] == [round(item) for item in orig_coords]:
                    encoding[self.sum_atom_types+self.atom_codes[89]] = 1.0 if plifs[coord][0] == 'hydrophobic' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[90]] = 1.0 if plifs[coord][0] == 'hbond' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[91]] = 1.0 if plifs[coord][0] == 'waterbridge' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[92]] = 1.0 if plifs[coord][0] == 'saltbridge' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[93]] = 1.0 if plifs[coord][0] == 'pistacking' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[94]] = 1.0 if plifs[coord][0] == 'pication' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[95]] = 1.0 if plifs[coord][0] == 'halogen' \
                    else 0.0
                    encoding[self.sum_atom_types+self.atom_codes[96]] = 1.0 if plifs[coord][0] == 'metal' \
                    else 0.0
                    
                    #distance
                    encoding[self.sum_atom_types+self.atom_codes[97]] = plifs[coord][1]

        
        return encoding
    
    #Get atom coords and atom features from the complexes.   
    def get_features(self, molecule, plifs, molprotein):
        coords = []
        features = []
            
        for atom in molecule:
            coords.append(atom.coords)
            if atom.atomicnum in [6,7,16]:
                atomicnum = (atom.atomicnum,atom.hyb)
                features.append(self.encode(atomicnum,atom.coords,plifs,molprotein))
            else:
                features.append(self.encode(atom.atomicnum,atom.coords,plifs,molprotein))
        
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)

        return coords, features
     
    #Define the rotation matrixs of 3D stuctures.
    def rotation_matrix(self, t, roller):
        if roller==0:
            return np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]])
        elif roller==1:
            return np.array([[np.cos(t),0,-np.sin(t)],[0,1,0],[np.sin(t),0,np.cos(t)]])
        elif roller==2:
            return np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])

    #Generate 3d grid or 4d tensor. Each grid represents a voxel. Each voxel represents the atom in it by onehot encoding of atomic type.
    #Each complex in train set is rotated 9 times for data amplification.
    #The complexes in core set are not rotated. 
    #The default resolution is 20*20*20.
    def grid(self,grid, coords, features, frag_idx, resolution=1.0, max_dist=10.0,  rotation_bool=True, max_frag=10, rotations=9):
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  

        slider=frag_idx*20

        x=y=z=np.array(range(-10,10),dtype=np.float32)+0.5
        u=0
        for i in range(len(coords)):
            coord=coords[i]
            # add/subtract 10 from the center
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z)

            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                u+=1
                # get the position of the closest point to coordinate which is found inside the grid
                # append the features unto that slice

                grid[0,slider+np.argmin(tmpx),slider+np.argmin(tmpy),slider+np.argmin(tmpz)] += torch.tensor(features[i])
                
        if rotation_bool:
            for rotation_idx in range(rotations):
                theta = random.uniform(np.pi/18,np.pi/2)
                roller = random.randrange(3)
                coords = np.dot(coords, self.rotation_matrix(theta,roller))
                for i in range(len(coords)):
                    coord=coords[i]
                    tmpx=abs(coord[0]-x)
                    tmpy=abs(coord[1]-y)
                    tmpz=abs(coord[2]-z)
                    if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                        grid[rotation_idx+1,slider+np.argmin(tmpx),slider+np.argmin(tmpy),slider+np.argmin(tmpz)] += features[i]

        return grid
    
class PLIF:
    def __init__(self, PDB: str, MOL_SPLIT_START: int = 70, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PLIF,self).__init__()
        
        self.MOL_SPLIT_START=MOL_SPLIT_START
        self.pdb=PDB
        self.records=['ATOM']
        self.values=['HOH','CL','MG','ZN','MN','CA']
        self.ions=['CL','MG','ZN','MN','CA']
        self.interaction_slices={"hydrophobic":[0,1,6,7,8,9,10],
            "hbond":[0,1,7,11,13,15,16],
            "waterbridge":[0,1,[6,7],11,13,16,17],
            "saltbridge":[0,1,7,10,3,11,12],
            "pistacking":[0,1,7,11,6,12,13],
            "pication":[0,1,7,11,3,12,13],
            "halogen":[0,1,7,10,12,14,15],
            "metal":[0,1,11,8,6,17,16]} 

        self.column_names = ['RESNR', 'RESTYPE', 'DIST', 'LIG_IDX','PROT_IDX','FRAGMENT_ATOMS_COORDS', 'AA_COORDS']
        self.path = os.getcwd()


    def okToBreak(self, bond):
        """
        Here we apply a bunch of rules to judge if the bond is OK to break.

        Parameters
        ----------
        bond :
            RDkit MOL object

        Returns
        -------
        Boolean :
            OK or not to break.
        """
        # See if the bond is in Ring (don't break that)
        if bond.IsInRing():
            return False
        # We OK only single bonds to break
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            return False

        # Get the beginning atom of the bond
        begin_atom = bond.GetBeginAtom()
        # Get the ending atom of the bond
        end_atom = bond.GetEndAtom()
        # What kind of neighbors does these end and begenning atoms have? We need a family of no less than 5!
        neighbor_end=list(end_atom.GetNeighbors())
        neighbor_begin=list(begin_atom.GetNeighbors())
        if (len(neighbor_end) + len(neighbor_begin)) <5:
            return False
        #for atm in neighbor_end:
            #print(atm.GetAtomicNum())
        #print(begin_atom.GetAtomicNum(), end_atom.GetAtomicNum(), MOL_SPLIT_START)
        
        # Now check if end or begenning atoms are in ring (we dont wanna bother those)
        if not(begin_atom.IsInRing() or end_atom.IsInRing()):
            return False
        elif begin_atom.GetAtomicNum() >= self.MOL_SPLIT_START or \
                end_atom.GetAtomicNum() >= self.MOL_SPLIT_START:
            return False
        elif end_atom.GetAtomicNum() == 1:
            return False
        else:
            return True

    def undo_id_label (self, frag, split_id):
        # I am trying to restore Hydrogens where the break happened
        for i, atom in enumerate(frag.GetAtoms()):
            if atom.GetAtomicNum() >= split_id:
                atom.SetAtomicNum(1)

        return frag

    # Divide a molecule into fragments
    def split_molecule(self, mol, pdb):

        split_id = self.MOL_SPLIT_START

        res = []
        res_no_id=[]

        to_check = [mol]
        while len(to_check) > 0:
            ms = self.spf(to_check.pop(), split_id)
            if len(ms) == 1:
                res += ms
            else:
                to_check += ms
                split_id += 1
        for frag in res:
            res_no_id.append(self.undo_id_label(frag, self.MOL_SPLIT_START))

        res_pdb_frags=[]

        for idx, frag in enumerate(res_no_id):
            w = Chem.PDBWriter(f"tmp_{pdb}_{self.MOL_SPLIT_START+idx}.pdb")
            w.write(frag)
            w.close()
            
            unwanted_entries= ['CONECT', 'END']            
            with open(f"tmp_{pdb}_{self.MOL_SPLIT_START+idx}.pdb") as oldfile, open(f"{pdb}_{self.MOL_SPLIT_START+idx}.pdb", 'w') as newfile:
                for line in oldfile:
                    if not any(unwanted_entry in line for unwanted_entry in unwanted_entries):
                        newfile.write(line)

                    
            data = data2 = ""

            # Reading data from file1
            with open(f"ATOM_{pdb}.pdb") as fp:
                data = fp.read()

            # Reading data from file2

            with open(f"{pdb}_{self.MOL_SPLIT_START+idx}.pdb") as fp:
                data2 = fp.read()
            
            # Merging 2 files
            # To add the data of file2
            # from next line
            #data += "\n"
            data += data2
            
            with open(f"HOH_{pdb}.pdb") as fp:
                data3 = fp.read()
            data += data3

            with open (f"ATOM_{pdb}_{self.MOL_SPLIT_START+idx}.pdb", 'w') as fp:
                fp.write(data)
            res_pdb_frags.append(f"ATOM_{pdb}_{self.MOL_SPLIT_START+idx}.pdb")
        return res_pdb_frags #create_chain(res)


    # Function for doing all the nitty gritty splitting work.
    # loops over bonds until bonds get exhausted or bonds are ok to break, whichever comes first. If ok to break, then each
    # fragment needs to be checked individually again through the loop
    def spf(self, mol, split_id):

        bonds = mol.GetBonds()
        for i in range(len(bonds)):
            if self.okToBreak(bonds[i]):
                mol = Chem.FragmentOnBonds(mol, [i])
                # Dummy atoms are always added last
                n_at = mol.GetNumAtoms()
                print('Split ID', split_id)
                mol.GetAtomWithIdx(n_at-1).SetAtomicNum(split_id)
                mol.GetAtomWithIdx(n_at-2).SetAtomicNum(split_id)
                return Chem.rdmolops.GetMolFrags(mol, asMols=True)

        # If the molecule could not been split, return original molecule
        return [mol]
    #get_fragments(fragment_mols)

    def retreive_plip_interactions(self, pdb_file):
        """
        Retreives the interactions from PLIP.

        Parameters
        ----------
        pdb_file :
            The PDB file of the complex. 

        Returns
        -------
        dict :
            A dictionary of the binding sites and the interactions.
        """
        protlig = PDBComplex()   #instantiate the loader from PLIP
        protlig.load_pdb(pdb_file)   # load the pdb file
        for ligand in protlig.ligands:
            protlig.characterize_complex(ligand)   # find ligands and analyze interactions
        sites = {}
        # loop over binding sites
        for key, site in sorted(protlig.interaction_sets.items()):
            binding_site = BindingSiteReport(site)   # collect data about interactions
            # tuples of *_features and *_info will be converted to pandas DataFrame
            keys = (
                "hydrophobic",
                "hbond",
                "waterbridge",
                "saltbridge",
                "pistacking",
                "pication",
                "halogen",
                "metal"
            )
        # interactions is a dictionary which contains relevant information for each
        # of the possible interactions: hydrophobic, hbond, etc. in the considered
        # binding site. Each interaction contains a list with 
        # 1. the features of that interaction, e.g. for hydrophobic:
        # ('RES_number', 'RES_type', ..., 'LIG_coord', 'PROT_coord')
        # 2. information for each of these features, e.g. for hydrophobic
        # ('RES_number', 'RES_type', ..., 'LIG_coord', 'PROT_coord')

            interactions = {
                k: [getattr(binding_site, k + "_features")] + getattr(binding_site, k + "_info")
                for k in keys
            }
            sites[key] = interactions
        return sites

    def get_coords_prot(self, RESNR):
        ppdb = PandasPdb()
        ppdb.read_pdb(f"{self.pdb.split('.')[0]}_protein.pdb")
        only_protein=ppdb.df['ATOM']
        resnr_coords=[]
        for i in RESNR:
            resnr_coords.append(list(only_protein[only_protein['atom_number']==int(i)][['x_coord', 'y_coord', 'z_coord']].values[0]))
        return resnr_coords
    
    def interaction_df(self, split):

        all_interactions_df = pd.DataFrame()


        # We create the dictionary for the complex of interest:
        for idx, s in enumerate(split):

            pdb_id=s.split('.')[0]
            raw=pdb_id.split('_')[1]
            idx_frag=int(pdb_id.split('_')[2])
            interactions_by_site = self.retreive_plip_interactions(f"{pdb_id}.pdb")

            # Let’s see how many binding sites are detected:

    #         print(
    #             f"Number of binding sites detected in {pdb_id} : "
    #             f"{len(interactions_by_site)}\n"
    #             f"with {interactions_by_site.keys()}"
    #         )
            # In this case, the first binding site containing ligand 03P will be further investigated.
            index_of_selected_site = 0
            selected_site = list(interactions_by_site.keys())[index_of_selected_site]
            #print(selected_site)


            valid_types = [
                    "hydrophobic",
                    "hbond",
                    "waterbridge",
                    "saltbridge",
                    "pistacking",
                    "pication",
                    "halogen",
                    "metal",
                ]

            for _type in valid_types:
                output_df=self.create_df_from_binding_site(raw, interactions_by_site[selected_site], idx+self.MOL_SPLIT_START, selected_site,
                                                      interactions_by_site,
                                                      interaction_type=_type)
                all_interactions_df=all_interactions_df.append(output_df)
        all_interactions_df = all_interactions_df[all_interactions_df['RESNR'].notna()]
        all_interactions_df.to_csv(f"{self.path}/results_plifs/{raw}_plifs_and_properties.csv", index=False)
        return all_interactions_df


    # We can construct a pandas.DataFrame for a binding site and particular interaction type.

    def create_df_from_binding_site(self, raw, selected_site_interactions, fragment_idx, selected_site, 
                                    interactions_by_site, interaction_type="hbond"):
        """
        Creates a data frame from a binding site and interaction type.

        Parameters
        ----------
        selected_site_interactions : dict
            Precalculated interactions from PLIP for the selected site
        interaction_type : str
            The interaction type of interest (default set to hydrogen bonding).

        Returns
        -------
        pd.DataFrame :
            DataFrame with information retreived from PLIP.
        """
        # check if interaction type is valid:
        valid_types = [
            "hydrophobic",
            "hbond",
            "waterbridge",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
        ]


        if interaction_type not in valid_types:
            print("!!! Wrong interaction type specified. Hbond is chosen by default !!! \n")
            interaction_type = "hbond"

        def interaction_values(n):
            try:
                interactions=interactions_by_site[selected_site][interaction_type]
                if type(n) is list:
                    return [interactions[1:][x][i] for x in 
                        range(len(interactions[1:])) for i in n]
                else:
                    return [interactions[1:][x][n] for x in 
                        range(len(interactions[1:]))]
            except Exception:
                return None
            
        if interactions_by_site[selected_site][interaction_type][1:]:
            #print(list(map(interaction_values, self.interaction_slices[interaction_type])), self.column_names)
            selected_feats=list(map(interaction_values, self.interaction_slices[interaction_type]))
            #print(selected_feats)
            try: 
                if int(selected_feats[4])>int(selected_feats[3]):
                    selected_feats[3], selected_feats[4] = selected_feats[4], selected_feats[3]  
            except: 
                if int(any(selected_feats[4]))>int(any(selected_feats[3])):
                    selected_feats[3], selected_feats[4] = selected_feats[4], selected_feats[3] 
            df = pd.DataFrame(
                # data is stored AFTER the columns names
                [selected_feats],
                # column names are always the first element - we skipped that in the above - we are gonna use that for naming the df
                columns = self.column_names
            )

            df["INTERACTION_TYPE"]=interaction_type
            
            try:
                checked_coords=self.get_coords_prot(selected_feats[4][0].split(',') if ',' in selected_feats[4][0] \
                                                                   else selected_feats[4])
            except:
                checked_coords=selected_feats[6]
                
            df["AA_COORDS"]=[checked_coords]
                #[self.get_coords_prot(selected_feats[4].split(','))]
            df["FRAGMENT_ATOMS_COORDS"]=[selected_feats[5]]
                            #[self.get_coords_lig(selected_feats[3].split(','))]    
            df['FRAGMENT_ID']=fragment_idx

            # ideally we would like to exclude waters from further processing. Threrfore let us reduce any waterbridge 
            # interaction to the eucladean distance in order to omit water
            
            if interaction_type == "waterbridge":
                df['DIST']=[[np.linalg.norm(x) for x in df['DIST'].to_numpy()]]
                
            # also deal with one distance value and two coords, this is common in saltbridge interactions:
            if len(checked_coords) == len(selected_feats[2])*2:
                df['DIST']=[selected_feats[2] + selected_feats[2]]
                
        else:

            df= pd.DataFrame({'RESNR':[None], 'RESTYPE':[None], 'DIST':[None], 'LIG_IDX':[None],'PROT_IDX':[None],
                        'INTERACTION_TYPE':[interaction_type], "AA_COORDS": [None], "FRAGMENT_ATOMS_COORDS":[None],
                              'FRAGMENT_ID':[str(fragment_idx)]})



        return df
    
    def pdb_2_sdf(self, pdb):
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "sdf")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, pdb)   # Open Babel will uncompress automatically

        mol.AddHydrogens()


        obConversion.WriteFile(mol, f"{pdb.split('.')[0]}.sdf")
        return f"{pdb.split('.')[0]}.sdf"
    
    def sdf_2_pdb(self, sdf):
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdb")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, sdf)   # Open Babel will uncompress automatically

        mol.AddHydrogens()
        obConversion.WriteFile(mol, f"{sdf.split('.')[0]}.pdb")
        return f"HETATM_{sdf.split('.')[0]}.pdb"

    def save_bpdb(self, pdb,ppdb, record):  
        ppdb.to_pdb(path=f"{record}_{pdb.split('.')[0].split('_')[0]}.pdb",
                    records=[record],
                    gz=False, 
                    append_newline=True)

    def get_HOH_pdb(self, pdb):
        ppdb = PandasPdb() 
        ppdb.read_pdb(pdb) 
        ppdb.df['HETATM']=ppdb.df['HETATM'].loc[ppdb.df['HETATM']['residue_name'].isin(self.values)]
        ppdb.to_pdb(path=f"HOH_{pdb.split('.')[0].split('_')[0]}.pdb",
                records=['HETATM'],
                gz=False, 
                append_newline=True)

    def keep_relevant_hetatm(self, pdb):
        raw=str(self.pdb).split('.')[0]
        with open(pdb) as f1, open(f"ATOM_{pdb.split('.')[0].split('_')[0]}.pdb", 'w') as f2:
            for line in f1:
                if 'ATOM' in line:
                    f2.write(line)
        with open(f'{raw}_ligand.pdb') as f1, open(f"HETATM_{pdb.split('.')[0].split('_')[0]}.pdb", 'w') as f2:
            for line in f1:
                if ('HETATM' in line) and not any(ion in line for ion in self.ions):
                    f2.write(line)
        try: 
            self.get_HOH_pdb(pdb)
        except:
            with open(pdb) as f1, open(f"HOH_{pdb.split('.')[0].split('_')[0]}.pdb", 'w') as f2:
                for line in f1:
                    if ('HETATM' in line) and any(ion in line for ion in self.ions):
                        f2.write(line)
        return
    
    
    def fragment_and_plif(self):
        path = os.getcwd()
        if not os.path.exists('results_plifs'):
            os.mkdir(f'{path}/results_plifs')

        raw=str(self.pdb).split('.')[0]
        self.sdf_2_pdb(f'{raw}_ligand.sdf')
        self.keep_relevant_hetatm(f'{raw}_protein.pdb')
        fragment_mols = Chem.SDMolSupplier(str(f'{raw}_ligand.sdf'), removeHs=True, sanitize=False)
        fragment_mols_alt = Chem.MolFromMol2File(f'{raw}_ligand.mol2', sanitize=True, removeHs=True)
        content = open(f'{raw}_ligand.pdb').read()
        hets=re.findall("^HETATM (.*)", content, re.M)
        if len(hets)<5:
            # Read in the file
            with open(f'{raw}_ligand.pdb', 'r') as file :
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace('ATOM  ', 'HETATM')

            # Write the file out again
            with open(f'{raw}_ligand.pdb', 'w') as file:
                file.write(filedata)
        
        try: 
            fragment_mols = Chem.RemoveHs(fragment_mols[0])
            output_df = self.interaction_df(self.split_molecule(fragment_mols,raw))
            
        except:  
            try:
                fragment_mols = Chem.SDMolSupplier(str(f'{raw}_ligand.sdf'), removeHs=True, sanitize=False)
                output_df = self.interaction_df(self.split_molecule(fragment_mols,raw))
            except:
                try: 
                    output_df = self.interaction_df(self.split_molecule( Chem.MolFromMol2File(fragment_mols_alt)))
                except:
                    try: 
                        fragment_mols = AllChem.MolFromPDBFile(f'{raw}_ligand.pdb')
                        output_df = self.interaction_df(self.split_molecule(fragment_mols,raw))
                    except:
                        try:
                            fragment_mols = AllChem.MolFromPDBFile(f'HETATM_{raw}.pdb')
                            output_df = self.interaction_df(self.split_molecule(fragment_mols,raw))
                        except:
    #                         raise Exception(f"Sorry, the pdb id: {raw} needs chekcing")
                            output_df= pd.DataFrame({'FRAGMENT_ID':[None], 'AA_COORDS':[None], 
                                                     'FRAGMENT_ATOMS_COORDS':[None], 
                                         'INTERACTION_TYPE':[None],'DIST':[None]})
                            print(f"Sorry, the pdb id: {raw} needs chekcing")
        os.chdir(f'{path}')

        return output_df.groupby('FRAGMENT_ID')['AA_COORDS', 'FRAGMENT_ATOMS_COORDS','INTERACTION_TYPE','DIST'].agg(list)


def kd_equalizer (value):

    if 'mM' in value.split('=')[1]:
        return float(value.split('m')[0].split('=')[1]) / 1000
    elif 'uM' in value.split('=')[1]:
        return float(value.split('u')[0].split('=')[1]) / 1000000
    elif 'nM' in value.split('=')[1]:
        return float(value.split('n')[0].split('=')[1]) / 1000000000
    elif 'pM' in value.split('=')[1]:
        return float(value.split('p')[0].split('=')[1]) / 1000000000000
    elif 'fM' in value.split('=')[1]:
        return float(value.split('f')[0].split('=')[1]) / 1000000000000000

def dask_plif_cnn_train(row_pdb, row_kd):
    train_grids=None
    train_label=[]
    pdb_id = row_pdb
    print('pdb_id', pdb_id)

    os.chdir(f'/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/general_refined_set/{pdb_id}')

    raw=pdb_id
    path = os.getcwd()
    fileList = []
    fileList.extend(glob.glob(f'{path}/{raw}_7*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_8*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_9*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*sdf'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    df_plifSpecs = PLIF(PDB = f'{pdb_id}.pdb').fragment_and_plif()
    
    if not len(df_plifSpecs) or df_plifSpecs['INTERACTION_TYPE'].apply(', '.join).unique()[0]=='metal':
        return

    train_label.extend([row_kd]*10)

    single_pdb_frags = []
    for idx, row in df_plifSpecs.iterrows():

        temp_plifs_prot={}
        temp_plifs_frag={}
        ## do something with fragment_idx . i.e. open the pdb and do your shit with encoding\
        for aa_atm_coord_list, frag_lig_atm_coord_list, interaction, dist_list in zip (row['AA_COORDS'], 
                                                                                           row['FRAGMENT_ATOMS_COORDS'],
                                                                                           row['INTERACTION_TYPE'],
                                                                                           row['DIST']):
            # because sometimes salt bridges makes two concurrent connections so it is possible that we have one distance
            # for two amino acids or ligand atoms! Encoding by atom is crazy fun
            for dist, aa_atm_coord, frag_lig_atm_coord in zip (dist_list, aa_atm_coord_list, 
                                                                             frag_lig_atm_coord_list):
                temp_plifs_prot[tuple(aa_atm_coord)]=[interaction,dist]
                temp_plifs_frag[tuple(frag_lig_atm_coord)]=[interaction,dist]


        pdb = next(pybel.readfile('pdb',os.path.join(path,'ATOM_' + pdb_id + '.pdb')))
        ligand = next(pybel.readfile('pdb',os.path.join(path, pdb_id + f'_{str(idx)}'+'.pdb')))
        single_pdb_frags.append((pdb,ligand,temp_plifs_prot,temp_plifs_frag))  

    grid=torch.zeros((full_batch,200,200,200,features_shape))

    for idx, mols in enumerate(single_pdb_frags):
        print(len(single_pdb_frags), idx)
#             threads = multiprocessing.Pool(len(single_pdb_frags))
#             threads.map(func, arg_list)

        coords1, features1 = Feature.get_features(mols[0],mols[2],1)
        coords2, features2 = Feature.get_features(mols[1],mols[3],0)

        # get the center point of protein
        center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
        coords=np.concatenate([coords1,coords2],axis = 0)
        features=np.concatenate([features1,features2],axis = 0)
        assert len(coords) == len(features)
        # zero the coordinates 
        coords = coords-center
        grid=Feature.grid(grid,coords,features,idx, rotation_bool=True)

    if train_grids is None:
        train_grids = grid.to_sparse()
    else:
        train_grids = torch.cat((train_grids,grid.to_sparse()), 0)
    print(train_grids.shape)

    raw=pdb_id
    path = os.getcwd()
    fileList = []
    fileList.extend(glob.glob(f'{path}/{raw}_7*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_8*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_9*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*sdf'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    

    print("Memory utilised (bytes): ", sys.getsizeof(train_grids))
    with open('train_grids.pkl','wb') as f:
        pickle.dump(train_grids, f)
        
    #Save the label data of training and test set.
    with open('train_label.pkl','wb') as f:
        pickle.dump(train_label,f)    
        
    os.chdir(p_directory)
    
    return pdb_id


def dask_plif_cnn_test(row_pdb, row_kd):
    
    test_grids=None
    test_label=[]
    pdb_id = row_pdb
    print('pdb_id', pdb_id)

    os.chdir(f'/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/general_refined_set/{pdb_id}')

    raw=pdb_id
    path = os.getcwd()
    fileList = []
    fileList.extend(glob.glob(f'{path}/{raw}_7*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_8*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_9*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*sdf'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    df_plifSpecs = PLIF(PDB = f'{pdb_id}.pdb').fragment_and_plif()
    
    if not len(df_plifSpecs):
        return
    
    test_label.extend([row_kd])

    single_pdb_frags = []
    print(len(single_pdb_frags))
    for idx, row in df_plifSpecs.iterrows():

        temp_plifs_prot={}
        temp_plifs_frag={}
        ## do something with fragment_idx . i.e. open the pdb and do your shit with encoding\
        for aa_atm_coord_list, frag_lig_atm_coord_list, interaction, dist_list in zip (row['AA_COORDS'], 
                                                                                           row['FRAGMENT_ATOMS_COORDS'],
                                                                                           row['INTERACTION_TYPE'],
                                                                                           row['DIST']):
            # because sometimes salt bridges makes two concurrent connections so it is possible that we have one distance
            # for two amino acids or ligand atoms! Encoding by atom is crazy fun
            for dist, aa_atm_coord, frag_lig_atm_coord in zip (dist_list, aa_atm_coord_list, 
                                                                             frag_lig_atm_coord_list):
                temp_plifs_prot[tuple(aa_atm_coord)]=[interaction,dist]
                temp_plifs_frag[tuple(frag_lig_atm_coord)]=[interaction,dist]


        pdb = next(pybel.readfile('pdb',os.path.join(path,'ATOM_' + pdb_id + '.pdb')))
        ligand = next(pybel.readfile('pdb',os.path.join(path, pdb_id + f'_{str(idx)}'+'.pdb')))
        single_pdb_frags.append((pdb,ligand,temp_plifs_prot,temp_plifs_frag))  

    grid=torch.zeros((1,200,200,200,features_shape))

    for idx, mols in enumerate(single_pdb_frags):
        coords1, features1 = Feature.get_features(mols[0],mols[2],1)
        coords2, features2 = Feature.get_features(mols[1],mols[3],0)

        # get the center point of protein
        center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
        coords=np.concatenate([coords1,coords2],axis = 0)
        features=np.concatenate([features1,features2],axis = 0)
        assert len(coords) == len(features)
        # zero the coordinates 
        coords = coords-center
        grid=Feature.grid(grid,coords,features,idx, rotation_bool=False)

    if test_grids is None:
        test_grids = grid.to_sparse()
    else:
        test_grids = torch.cat((test_grids,grid.to_sparse()), 0)
    print(test_grids.shape)

    raw=pdb_id
    path = os.getcwd()
    fileList = []
    fileList.extend(glob.glob(f'{path}/{raw}_7*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_8*pdb'))
    fileList.extend(glob.glob(f'{path}/{raw}_9*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*pdb'))
    fileList.extend(glob.glob(f'{path}/*_{raw}*sdf'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    


    with open('test_grids.pkl','wb') as f:
        pickle.dump(test_grids, f)
        
    #Save the label data of training and test set.
    with open('test_label.pkl','wb') as f:
        pickle.dump(test_label,f)    
        
    os.chdir(p_directory)
    
    return pdb_id

if __name__ == "__main__":
    
    os.chdir(f'/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/')

    Feature = Feature_extractor()

    p_directory = os.getcwd()

    dask_plif_cnn_test(str(sys.argv[1]), float(sys.argv[2]))

#     general=pd.read_csv('INDEX_general_PL_data.2020', sep=',')
#     refined=pd.read_csv('INDEX_refined_data.2020', sep=',')

#     general=general[general["Kd/Ki"].str.contains('IC|EC|>|<')==False]
#     refined=refined[refined["Kd/Ki"].str.contains('IC|EC|>|<')==False]

#     general["Kd/Ki"] = general["Kd/Ki"].str.replace('~','=')
#     refined["Kd/Ki"] = refined["Kd/Ki"].str.replace('~','=')


#     general['Kd/Ki']=general['Kd/Ki'].apply(lambda x: kd_equalizer(x))
#     refined['Kd/Ki']=refined['Kd/Ki'].apply(lambda x: kd_equalizer(x))

#     merged_PDBBind=general.append(refined) \
#                                 .sample(frac=1) \
#                                 .sample(frac=1) \
#                                 .reset_index(drop=True) \
#                                 .drop_duplicates(subset='PDB_code', keep="first") 

##    merged_PDBBind=pd.read_csv('merged_PDBBind.csv')
#     merged_PDBBind.rename(columns={'Kd/Ki': 'Kd_Ki'}, inplace=True)
#     merged_PDBBind[merged_PDBBind['PDB_code'].str.contains("3bho")==False]

##    train_df, test_df = train_test_split(merged_PDBBind, test_size=0.1)

#     merged_PDBBind.to_csv('merged_PDBBind.csv', index=False)
##    train_df.to_csv('train_df.csv', index=False)
##    test_df.to_csv('test_df.csv', index=False)


##    train_df.parallel_apply(lambda x: dask_plif_cnn_train(x.PDB_code, x.Kd_Ki), axis=1)
##    test_df.parallel_apply(lambda x: dask_plif_cnn_test(x.PDB_code, x.Kd_Ki), axis=1)

