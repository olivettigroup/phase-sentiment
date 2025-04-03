# Phase-sentiment
Repository accompanying paper for the extraction of phase-property relationships from metallurgical literature.

Montanelli, L., Venugopal, V., Olivetti, E.A. et al. High-Throughput Extraction of Phase–Property Relationships from Literature Using Natural Language Processing and Large Language Models. Integr Mater Manuf Innov 13, 396–405 (2024). https://doi.org/10.1007/s40192-024-00344-8

## Data
The final data used in the paper can be found in data/final_data.npy. 

For now, two BERT models are missing as their files are bigger than 100MB.

## List of requirements:
- numpy
- pandas
- json
- torch
- [BERTopic](https://maartengr.github.io/BERTopic/index.html)
