import os
from explainable_fact_checking import models
from explainable_fact_checking.experiment_definitions import C
from explainable_fact_checking.model_adapters.genFCExp import GenFCExp

input_list = [{
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 0,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 1,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 2,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 3,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 4,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 5,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 6,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 7,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 8,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect... an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 9,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 10,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 11,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 12,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 13,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 14,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 15,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 16,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 17,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 18,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 19,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 20,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 21,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 22,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 23,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 24,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...ith common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 25,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect... an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 26,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 27,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...re likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 28,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...ith common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 29,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...ith common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 30,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 31,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 32,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect... an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 33,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 34,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 35,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 36,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 37,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 38,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 39,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 40,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 41,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect... an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 42,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 43,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 44,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [' . '], 'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 45,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 46,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...ith common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 47,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...ith common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                          'In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 48,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect...d to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.',
                      'label': 2}, {
                      'claim': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.',
                      'evidence': [
                          'We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associa...s" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.',
                          'We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible ...etic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.',
                          'Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.',
                          'We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.',
                          'Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.',
                          'Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.'],
                      'goldtag': [1, 0, 0, 0, 1, 0, 0, 1], 'id': 49,
                      'input_txt_to_use': '1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effect... an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.',
                      'label': 2}]

def test_predict():
    model = GenFCExp(model_path=os.path.join(C.BASE_DIR_V2, 'models/Isabelle/scifact/isabelle_k20_mink5.pt'))
    out = model.predict(input_list, return_exp=True)
    assert len(out[0]) == len(input_list)
    assert [len(x['evidence']) for x in input_list] == [len(x) for x in out[1]]
