import requests

def fetch_bacteria_sequences(taxon_id, rettype="gb", retmode="text", seq_type="nucleotide"):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": seq_type,
        "id": taxon_id,
        "rettype": rettype,
        "retmode": retmode
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code}")
        return None

# Example: Fetch cDNA (nucleotide) sequences for Escherichia coli (taxon ID: 562)
bacteria_taxon_id = 562
cDNA_sequence = fetch_bacteria_sequences(bacteria_taxon_id)
print(cDNA_sequence)