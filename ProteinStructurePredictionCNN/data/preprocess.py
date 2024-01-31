"""
!pip install biopython
"""
from Bio import SeqIO
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.SASA import ShrakeRupley
from Bio import AlignIO
from collections import Counter

''' BLOSUM matrices '''
blosum62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,

    ('R', 'A'): -1, ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3,
    ('R', 'Q'): 1, ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3,
    ('R', 'L'): -2, ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2,
    ('R', 'S'): -1, ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,

    ('N', 'A'): -2, ('N', 'R'): 0, ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3,
    ('N', 'Q'): 0, ('N', 'E'): 0, ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3,
    ('N', 'L'): -3, ('N', 'K'): 0, ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2,
    ('N', 'S'): 1, ('N', 'T'): 0, ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,

    ('D', 'A'): -2, ('D', 'R'): -2, ('D', 'N'): 1, ('D', 'D'): 6, ('D', 'C'): -3,
    ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1, ('D', 'H'): -1, ('D', 'I'): -3,
    ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3, ('D', 'F'): -3, ('D', 'P'): -1,
    ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4, ('D', 'Y'): -3, ('D', 'V'): -3,

    ('C', 'A'): 0, ('C', 'R'): -3, ('C', 'N'): -3, ('C', 'D'): -3, ('C', 'C'): 9,
    ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3, ('C', 'I'): -1,
    ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2, ('C', 'P'): -3,
    ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2, ('C', 'V'): -1,

    ('Q', 'A'): -1, ('Q', 'R'): 1, ('Q', 'N'): 0, ('Q', 'D'): 0, ('Q', 'C'): -3,
    ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0, ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,

    ('E', 'A'): -1, ('E', 'R'): 0, ('E', 'N'): 0, ('E', 'D'): 2, ('E', 'C'): -4,
    ('E', 'Q'): 2, ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3,
    ('E', 'L'): -3, ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1,
    ('E', 'S'): 0, ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,

    ('G', 'A'): 0, ('G', 'R'): -2, ('G', 'N'): 0, ('G', 'D'): -1, ('G', 'C'): -3,
    ('G', 'Q'): -2, ('G', 'E'): -2, ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4,
    ('G', 'L'): -4, ('G', 'K'): -2, ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2,
    ('G', 'S'): 0, ('G', 'T'): -2, ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,

    ('H', 'A'): -2, ('H', 'R'): 0, ('H', 'N'): 1, ('H', 'D'): -1, ('H', 'C'): -3,
    ('H', 'Q'): 0, ('H', 'E'): 0, ('H', 'G'): -2, ('H', 'H'): 8, ('H', 'I'): -3,
    ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2, ('H', 'F'): -1, ('H', 'P'): -2,
    ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): -2, ('H', 'Y'): 2, ('H', 'V'): -3,

    ('I', 'A'): -1, ('I', 'R'): -3, ('I', 'N'): -3, ('I', 'D'): -3, ('I', 'C'): -1,
    ('I', 'Q'): -3, ('I', 'E'): -3, ('I', 'G'): -4, ('I', 'H'): -3, ('I', 'I'): 4,
    ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1, ('I', 'F'): 0, ('I', 'P'): -3,
    ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1, ('I', 'V'): 3,

    ('L', 'A'): -1, ('L', 'R'): -2, ('L', 'N'): -3, ('L', 'D'): -4, ('L', 'C'): -1,
    ('L', 'Q'): -2, ('L', 'E'): -3, ('L', 'G'): -4, ('L', 'H'): -3, ('L', 'I'): 2,
    ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,

    ('K', 'A'): -1, ('K', 'R'): 2, ('K', 'N'): 0, ('K', 'D'): -1, ('K', 'C'): -3,
    ('K', 'Q'): 1, ('K', 'E'): 1, ('K', 'G'): -2, ('K', 'H'): -1, ('K', 'I'): -3,
    ('K', 'L'): -2, ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1,
    ('K', 'S'): 0, ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,

    ('M', 'A'): -1, ('M', 'R'): -1, ('M', 'N'): -2, ('M', 'D'): -3, ('M', 'C'): -1,
    ('M', 'Q'): 0, ('M', 'E'): -2, ('M', 'G'): -3, ('M', 'H'): -2, ('M', 'I'): 1,
    ('M', 'L'): 2, ('M', 'K'): -1, ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2,
    ('M', 'S'): -1, ('M', 'T'): -1, ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,

    ('F', 'A'): -2, ('F', 'R'): -3, ('F', 'N'): -3, ('F', 'D'): -3, ('F', 'C'): -2,
    ('F', 'Q'): -3, ('F', 'E'): -3, ('F', 'G'): -3, ('F', 'H'): -1, ('F', 'I'): 0,
    ('F', 'L'): 0, ('F', 'K'): -3, ('F', 'M'): 0, ('F', 'F'): 6, ('F', 'P'): -4,
    ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1, ('F', 'Y'): 3, ('F', 'V'): -1,

    ('P', 'A'): -1, ('P', 'R'): -2, ('P', 'N'): -2, ('P', 'D'): -1, ('P', 'C'): -3,
    ('P', 'Q'): -1, ('P', 'E'): -1, ('P', 'G'): -2, ('P', 'H'): -2, ('P', 'I'): -3,
    ('P', 'L'): -3, ('P', 'K'): -1, ('P', 'M'): -2, ('P', 'F'): -4, ('P', 'P'): 7,
    ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3, ('P', 'V'): -2,

    ('S', 'A'): 1, ('S', 'R'): -1, ('S', 'N'): 1, ('S', 'D'): 0, ('S', 'C'): -1,
    ('S', 'Q'): 0, ('S', 'E'): 0, ('S', 'G'): 0, ('S', 'H'): -1, ('S', 'I'): -2,
    ('S', 'L'): -2, ('S', 'K'): 0, ('S', 'M'): -1, ('S', 'F'): -2, ('S', 'P'): -1,
    ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,

    ('T', 'A'): 0, ('T', 'R'): -1, ('T', 'N'): 0, ('T', 'D'): -1, ('T', 'C'): -1,
    ('T', 'Q'): -1, ('T', 'E'): -1, ('T', 'G'): -2, ('T', 'H'): -2, ('T', 'I'): -1,
    ('T', 'L'): -1, ('T', 'K'): -1, ('T', 'M'): -1, ('T', 'F'): -2, ('T', 'P'): -1,
    ('T', 'S'): 1, ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,

    ('W', 'A'): -3, ('W', 'R'): -3, ('W', 'N'): -4, ('W', 'D'): -4, ('W', 'C'): -2,
    ('W', 'Q'): -2, ('W', 'E'): -3, ('W', 'G'): -2, ('W', 'H'): -2, ('W', 'I'): -3,
    ('W', 'L'): -2, ('W', 'K'): -3, ('W', 'M'): -1, ('W', 'F'): 1, ('W', 'P'): -4,
    ('W', 'S'): -3, ('W', 'T'): -2, ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,

    ('Y', 'A'): -2, ('Y', 'R'): -2, ('Y', 'N'): -2, ('Y', 'D'): -3, ('Y', 'C'): -2,
    ('Y', 'Q'): -1, ('Y', 'E'): -2, ('Y', 'G'): -3, ('Y', 'H'): 2, ('Y', 'I'): -1,
    ('Y', 'L'): -1, ('Y', 'K'): -2, ('Y', 'M'): -1, ('Y', 'F'): 3, ('Y', 'P'): -3,
    ('Y', 'S'): -2, ('Y', 'T'): -2, ('Y', 'W'): 2, ('Y', 'Y'): 7, ('Y', 'V'): -1,

    ('V', 'A'): 0, ('V', 'R'): -3, ('V', 'N'): -3, ('V', 'D'): -3, ('V', 'C'): -1,
    ('V', 'Q'): -2, ('V', 'E'): -2, ('V', 'G'): -3, ('V', 'H'): -3, ('V', 'I'): 3,
    ('V', 'L'): 1, ('V', 'K'): -2, ('V', 'M'): 1, ('V', 'F'): -1, ('V', 'P'): -2,
    ('V', 'S'): -2, ('V', 'T'): 0, ('V', 'W'): -3, ('V', 'Y'): -1, ('V', 'V'): 4
}


    
def PSSM(fasta_file_path, amino_acids_order = 'ACEDGFIHKMLNQPSRTWVY-'):
    """
    Creating a position-specific scoring system (PSSM) in Python from multiple sequence alignments (MSAs). PSSMs will help with sequence analysis and motif discovery. 
    """
    # Load the MSA (multiple sequence alignment) from a FASTA file
    alignment = AlignIO.read(fasta_file_path, "fasta")
    position_specific_frequencies = []

    for i in range(len(alignment[0])):
        column = alignment[:, i]
        aa_frequencies = Counter(column)
        position_specific_frequencies.append(aa_frequencies)
        
    pssm_scores = []

    for frequencies in position_specific_frequencies:
        position_scores = {}
        for aa, frequency in frequencies.items():
            if aa in blosum62:
                position_scores[aa] = frequency * blosum62[aa]
        pssm_scores.append(position_scores)
        
    # Define the order of amino acids and initialize the scores
    default_score = 0

    # Create a list to store the position-specific scores in the specified order
    position_scores_list = [scores_at_position.get(aa, default_score) for aa in amino_acids_order]
    return position_scores_list

    

''' Secondary Structure of Proteins '''
def predict_DSSP(pdb_file_path):
    
    # Load the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    # Create a DSSP object to predict secondary structure
    dssp = DSSP(structure[0], pdb_file_path)

    # Get DSSP data for each residue in the protein
    ss_list, exp_list = [], []
    for residue_data in dssp:
        residue_id, (aa, ss, exp) = residue_data
        # print(f'Residue {residue_id}: AA={aa}, DSSP Secondary Structure={ss}, DSSP Exposition={exp}')
        ss_list.append(ss)
        exp_list.append(exp)
    return ss_list+exp_list

''' Calculation of solvent accessible surface areas '''
def predict_SASA(pdb_file_path):
    """
    probe_radius=1.3
    Absolute accessibility values with probe radius 1.3 (roughly the radius of water molecule)
    Relative accessibility values are normalized by the largest accessibility value in a protein.
    """
    
    p = PDBParser(QUIET=1)

    # Load the protein structure from a PDB file
    struct = p.get_structure("ex", pdb_file_path)

    # Create a ShrakeRupley object to calculate ASA
    sr = ShrakeRupley(probe_radius=1.3)

    # Compute ASA for the entire structure
    sr.compute(struct, level="S")

    # Calculate the total absolute solvent accessibility (ASA) for the structure
    total_asa = round(struct.sasa, 2)
    
    sr.compute(struct, level="R")
    
    sr.compute(ch1_struct[1], level="R")
    max_sasa = -10000.

    for res in struct:
        max_sasa = max(max_sasa,round(res.sasa,2))
        
    return total_asa, total_asa/max_sasa #Absolute, Relative accessibility

    
        
      