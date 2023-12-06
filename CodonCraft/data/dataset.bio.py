from Bio import Entrez, SeqIO
import ssl
import os

current_directory = os.path.dirname(__file__)
raw_data_fld = f"{current_directory}/raw/"


def save_file(filename, records):
    for i, record in enumerate(records):
        filename = f'{raw_data_fld}mrna_{i + 1}.fasta'
        with open(filename, 'w') as output_file:
            SeqIO.write(record, output_file, 'fasta')
        print(f"mRNA sequence {i + 1} saved to {filename}")

def fetch_protein_and_mrna_sequences(taxon_id='Escherichia coli', limit=10, restart=1):
    # Disable SSL certificate verification (not recommended for production)
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Set your email address for Entrez
    Entrez.email = "xyz@google.com"

    # Search for Escherichia coli mRNA sequences
    mrna_sequences, protein_sequences = [], []
    mrna_query = f"{taxon_id}[Organism]+AND+biomol+mrna[prop]"
    mrna_handle = Entrez.esearch(db='nucleotide', term=mrna_query, retmax=limit,restart=restart)  # Adjust retmax as needed
    mrna_record = Entrez.read(mrna_handle)
    mrna_ids = mrna_record['IdList']
    print (mrna_ids)
    # # Fetch mRNA sequences
    # mrna_sequences = []
    # for mrna_id in mrna_ids:
    #     mrna_handle = Entrez.efetch(db='nucleotide', id=mrna_id, rettype='fasta', retmode='text')
    #     mrna_record = SeqIO.read(mrna_handle, 'fasta')
    #     mrna_sequences.append(mrna_record)

    # # Search for Escherichia coli protein (amino acid) sequences
    # protein_query = f"{taxon_id}[Organism]+AND+biomol+protein[prop]"
    # protein_handle = Entrez.esearch(db='protein', term=protein_query, retmax=limit)  # Adjust retmax as needed
    # protein_record = Entrez.read(protein_handle)
    # protein_ids = protein_record['IdList']

    # # Fetch protein sequences
    # protein_sequences = []
    # for protein_id in protein_ids:
    #     protein_handle = Entrez.efetch(db='protein', id=protein_id, rettype='fasta', retmode='text')
    #     protein_record = SeqIO.read(protein_handle, 'fasta')
    #     protein_sequences.append(protein_record)

    return mrna_sequences, protein_sequences

if __name__ == "__main__":
    ecoli_mrna_seqs, ecoli_protein_seqs = fetch_protein_and_mrna_sequences()
    # Print or further process the fetched sequences
    print("mRNA Sequences:", ecoli_mrna_seqs)
    for mrna_seq in ecoli_mrna_seqs:
        print(f">{mrna_seq.id}\n{mrna_seq.seq}\n")

    print("Protein Sequences:")
    for protein_seq in ecoli_protein_seqs:
        print(f">{protein_seq.id}\n{protein_seq.seq}\n")