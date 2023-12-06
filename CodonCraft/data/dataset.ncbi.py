import requests
import warnings
import os

warnings.filterwarnings("ignore")
current_directory = os.path.dirname(__file__)
raw_data_fld = f"{current_directory}/raw/"

def assert_response(response):
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code}")
        return None
    
def test_ncbi_api():
    """https://www.ncbi.nlm.nih.gov/books/NBK25500/#chapter1.Downloading_Full_Records
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=34577062,24475906&rettype=fasta&retmode=text"
    response = requests.post(base_url, verify=False)
    
    return assert_response(response)
    

def retrieve_ecoli_sequences(ecoli_taxon_id = 'Escherichia coli', limit=100):
      # Escherichia coli taxon ID
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

    # Step 1: ESearch to retrieve GI numbers and post them on the History server for mRNA
    mRNA_query = f'{ecoli_taxon_id}[orgn]+AND+biomol+mrna[prop]'
    esearch_url_mRNA = f'{base_url}esearch.fcgi?db=nucleotide&term={mRNA_query}&usehistory=y'
    esearch_response_mRNA = requests.get(esearch_url_mRNA, verify=False)
    esearch_response_mRNA.raise_for_status()

    web_env_mRNA = esearch_response_mRNA.text.split('<WebEnv>')[1].split('</WebEnv>')[0]
    query_key_mRNA = esearch_response_mRNA.text.split('<QueryKey>')[1].split('</QueryKey>')[0]
    count_mRNA = esearch_response_mRNA.text.split('<Count>')[1].split('</Count>')[0]

    # Step 2: EFetch to retrieve mRNA data in batches
    retmax_mRNA = min(limit, int(count_mRNA))
    with open('ecoli_mRNA.fna', 'w') as output_file_mRNA:
        for retstart_mRNA in range(0, retmax_mRNA, limit):
            efetch_url_mRNA = f'{base_url}efetch.fcgi?db=nucleotide&WebEnv={web_env_mRNA}'
            efetch_url_mRNA += f'&query_key={query_key_mRNA}&retstart={retstart_mRNA}'
            efetch_url_mRNA += f'&retmax={limit}&rettype=fasta&retmode=text'
            efetch_response_mRNA = requests.get(efetch_url_mRNA, verify=False)
            efetch_response_mRNA.raise_for_status()
            output_file_mRNA.write(efetch_response_mRNA.text)

    # Step 3: EFetch to retrieve amino acid data (protein) in batches
    protein_query = f'{ecoli_taxon_id}[orgn]+AND+biomol+protein[prop]'
    esearch_url_protein = f'{base_url}esearch.fcgi?db=protein&term={protein_query}&usehistory=y'
    esearch_response_protein = requests.get(esearch_url_protein, verify=False)
    esearch_response_protein.raise_for_status()

    web_env_protein = esearch_response_protein.text.split('<WebEnv>')[1].split('</WebEnv>')[0]
    query_key_protein = esearch_response_protein.text.split('<QueryKey>')[1].split('</QueryKey>')[0]
    count_protein = esearch_response_protein.text.split('<Count>')[1].split('</Count>')[0]

    retmax_protein = min(limit, int(count_protein))
    print(f"Total protein sequences: {count_protein}")

    with open('ecoli_protein.faa', 'w') as output_file_protein:
        for retstart_protein in range(0, retmax_protein, limit):
            efetch_url_protein = f'{base_url}efetch.fcgi?db=protein&WebEnv={web_env_protein}'
            efetch_url_protein += f'&query_key={query_key_protein}&retstart={retstart_protein}'
            efetch_url_protein += f'&retmax={limit}&rettype=fasta&retmode=text'
            print(f"EFetch URL for protein: {efetch_url_protein}")
            efetch_response_protein = requests.get(efetch_url_protein, verify=False)
            efetch_response_protein.raise_for_status()
            
            # Check if the response contains data
            if efetch_response_protein.text.strip():
                output_file_protein.write(efetch_response_protein.text)
            else:
                print(f"No protein sequences found in this batch.")



if __name__ == "__main__":
    retrieve_ecoli_sequences(limit=100)

    
# # Example: Fetch cDNA (nucleotide) sequences for Escherichia coli (taxon ID: 562)
# bacteria_taxon_id = 562
# cDNA_sequence = fetch_bacteria_sequences(bacteria_taxon_id)
# print(cDNA_sequence)
