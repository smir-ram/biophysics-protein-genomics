This dataset provides information on various features associated with protein sequences. The dataset is designed to facilitate research and analysis in the field of bioinformatics and protein structure prediction. It contains 25 features + 8 labels, each providing valuable information about protein sequences.

# Features
The features are categorized as follows:

## N- and C-Terminals (Features 0 to 1)
These features provide information about the N- and C-terminals of protein sequences.
They offer insights into the terminus regions of the proteins.
## Solvent Accessibility (Features 2 to 3)
These features include both relative and absolute solvent accessibility values. Absolute accessibility values with probe radius 1.3 (roughly the radius of water molecule), while relative accessibility values are normalized by the largest accessibility value in a protein.

(The original solvent accessibility values are computed using the DSSP method)

## PSSM - position-specific scoring systems (Features 4 to 25)
The sequence profile features provide detailed information about the occurrence of amino acid residues.
The order of amino acid residues in the sequence profile is `ACDEFGHIKLMNPQRSTVWXY`. These features offer a comprehensive view (evolutionary neighbourhood - from multiple sequence alignments x BLOSUM62) of the amino acid composition within protein sequences.
These features represent amino acid residues found in protein sequences as follows -

A: Alanine
C: Cysteine
E: Glutamic Acid
D: Aspartic Acid
G: Glycine
F: Phenylalanine
I: Isoleucine
H: Histidine
K: Lysine
M: Methionine
L: Leucine
N: Asparagine
Q: Glutamine
P: Proline
S: Serine
R: Arginine
T: Threonine
W: Tryptophan
V: Valine
X: Represents "synthetic/man-made" amino acid, ion/ligand/small-molecule
Y: Tyrosine

# Labels
These features represent secondary structure labels associated with protein sequences.

## Secondary Structure Labels (Features 26 to 33)
The sequence of secondary structure labels includes: 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'.

label representations are as follows:
> L (Loop): 
This label is used to denote regions in a protein where the backbone does not form a regular secondary structure element like an alpha helix or beta sheet. Loops are often flexible and connect different secondary structure elements.

> B (Bridge): 
The 'B' label is used to represent residues that are part of a beta bridge. Beta bridges occur when two beta strands are connected by a hydrogen bond.

> E (Extended): 
This label is assigned to residues in extended or beta-sheet-like conformation. Beta sheets are formed when multiple strands of the protein backbone align and form hydrogen bonds.

>G (3-10 Helix): 
The 'G' label represents residues that are part of a 3-10 helix. The 3-10 helix is a type of secondary structure that forms when the protein backbone forms a helical shape with three residues per turn.

>I (Pi-Helix):
The 'I' label denotes residues in a pi-helix. Pi-helices are a less common secondary structure element where the protein backbone forms a helical shape with four residues per turn.

>H (Alpha Helix): 
The 'H' label is used for residues in an alpha helix. Alpha helices are a common secondary structure element in which the protein backbone forms a helical shape with 3.6 residues per turn.

>S (Bend): 
The 'S' label represents residues in a bend or a region where the protein backbone changes direction or orientation. Bends are often found at the junctions between different secondary structure elements.

>T (Turn): 
The 'T' label is assigned to residues in a turn or loop that connects two strands of a beta sheet. Turns facilitate the folding of beta sheets.




# Usage
Commonly utilized dataset for various tasks, including protein structure prediction, secondary structure analysis, and sequence profiling. The dataset's diverse set of features provides valuable information for protein-related studies.

# Citation

1. CullPDB53 Dataset (6125 proteins):The CullPDB53 dataset is a non-redundant set of protein structures from the Protein Data Bank (PDB). https://www.rcsb.org/.

2. The CB513 dataset is often used for protein secondary structure prediction. https://www.princeton.edu/~jzthree/datasets/ICML2014/.

3. The Critical Assessment of Structure Prediction (CASP) datasets are used for protein structure prediction and related tasks. http://predictioncenter.org/.

4. CAMEO Test Proteins (6 months): The CAMEO (Continuous Automated Model EvaluatiOn) test proteins are used for protein structure prediction evaluation. http://www.cameo3d.org/sp/6-months/.

5. JPRED Training and Test Data (1338 training and 149 test proteins): The JPRED dataset provides training and test data for protein secondary structure prediction. http://www.compbio.dundee.ac.uk/jpred4/about.shtml.


