# DeepMEP
A deep learning model that can quantify the missense mutation effect on phosphorylation.

## Phosphorylation annotations
The phosphorylation annotations should be placed at directory as followes:
1. CBMAP: 'data/dataset/CBMAP/lab_phos_data.txt'
2. Uniprot: 'data/dataset/uniprot/uniprot_phos_data.txt'
3. EPSD: 'data/dataset/EPSD_annotation/phos_kinase_anno.txt'

### Phosphorylation intensity in CBMAP
All phosphorylation annotations from CBMAP should be processed into the table as followes:
| accession      | pos | aa | standard_log_mean_intensity  |
|----------------|-----|----|------------------------------|
| A0A087WUV0     | 121 | S  | -0.302                       |
| A0A0A0MRZ7     | 100 | S  | -0.061                       |
| A0A0B4J2A2     | 98  | S  | 0.305                        |
| A0A0B4J2A2     | 67  | T  | -0.307                       |
| A0A0B4J2D5     | 108 | S  | -0.808                       |
| ...            | ... | ...| ...                          |
| Q9Y6Y8         | 929 | S  | -0.275                       |
| Q9Y6Y8         | 993 | S  | -0.757                       |
| Q9Y6Y8         | 331 | Y  | 1.780                        |
| Q9Y6Y8         | 934 | Y  | -0.824                       |
| W5XKT8         | 278 | S  | 0.546                        |

### General phosphorylation annotation in Uniprot
All phosphorylated sites (non-phosphorylated sites are not required) from Uniprot should be processed into the table as followes:
| accession | pos | aa | phosphorylation |
|-----------|-----|----|-----------------|
| A0AVK6    | 70  | S  | 1               |
| A0AVK6    | 101 | S  | 1               |
| A0AVK6    | 412 | S  | 1               |
| A0AVK6    | 416 | S  | 1               |
| A0AVT1    | 53  | T  | 1               |
| ...       | ... | ...| ...             |
| Q9Y546    | 405 | S  | 1               |
| Q9Y546    | 406 | S  | 1               |
| Q5VUE5    | 66  | S  | 1               |
| Q8IXQ3    | 68  | S  | 1               |
| Q8IXQ3    | 75  | S  | 1               |

### Kinase-specific phosphorylation annotation in EPSD
All kinase-specific phosphorylation annotations from EPSD should be processed into the table as followes:
| accession | pos | aa | kinase    | db            | Name   | Group | Family | Kinase Domain                                                                 |
|-----------|-----|----|-----------|---------------|--------|-------|--------|-------------------------------------------------------------------------------|
| P05198    | 51  | S  | HRI       | PhosphoSitePlus | HRI    | OTHER | PEK    | FEELAILGKGGYGRVYKVRNKLDGQYYAIKKILIKGATKTVCMKVL...                           |
| P05198    | 48  | S  | HRI       | PhosphoSitePlus | HRI    | OTHER | PEK    | FEELAILGKGGYGRVYKVRNKLDGQYYAIKKILIKGATKTVCMKVL...                           |
| Q9UQL6    | 258 | S  | PKCD      | PhosphoSitePlus | PKCD   | AGC   | PKC    | FIFHKVLGKGSFGKVLLGELKGRGEYFAIKALKKDVVLIDDDVECT...                           |
| P61978    | 301 | S  | PKCD      | PhosphoSitePlus | PKCD   | AGC   | PKC    | FIFHKVLGKGSFGKVLLGELKGRGEYFAIKALKKDVVLIDDDVECT...                           |
| Q9UQ13    | 70  | T  | PKCD      | PhosphoSitePlus | PKCD   | AGC   | PKC    | FIFHKVLGKGSFGKVLLGELKGRGEYFAIKALKKDVVLIDDDVECT...                           |
| ...       | ... | ...| ...       | ...             | ...    | ...   | ...    | ...                                                                         |
| Q96L34    | 213 | T  | MARK4     | RegPhos         | MARK4  | CAMK  | CAMKL  | YRLLRTIGKGNFAKVKLARHILTGREVAIKIIDKTQLNPSSLQKLF...                           |
| Q99759    | 525 | S  | MAP3K3    | RegPhos         | MAP3K3 | STE   | STE11  | WRRGKLLGQGAFGRVYLCYDVDTGRELASKQVQFDPDSPETSKEVS...                           |
| Q9H2K8    | 180 | T  | SLK       | RegPhos         | SLK    | STE   | STE20  | WEIIGELGDGAFGKVYKAQNKETSVLAAAKVIDTKSEEELEDYMVE...                           |
| Q9H2K8    | 182 | Y  | SLK       | RegPhos         | SLK    | STE   | STE20  | WEIIGELGDGAFGKVYKAQNKETSVLAAAKVIDTKSEEELEDYMVE...                           |
| O00303    | 45  | S  | PITSLRE   | PSEA            | PITSLRE| CMGC  | CDK    | FQCLNRIEEGTYGVVYRAKDKKTDEIVALKRLKMEKEKEGFPITSL...                           |

## SNP annotations
### missense mutation
The mutant sequence and reference sequence for each missense should be processed into the table as followes:
| variant_info                  | transtript_id   | cds_variant   | aa_variant   | pos | aa_ref | aa_alt | names                                                            | width | SNP          | ref | alt | gene   | sequence                                                                 |
|-------------------------------|-----------------|---------------|--------------|-----|--------|--------|------------------------------------------------------------------|-------|--------------|-----|-----|--------|--------------------------------------------------------------------------|
| NM_000015:c.A766G:p.K256E     | NM_000015       | c.A766G       | p.K256E      | 256 | K      | E      | line21115543 NM_000015 c.A766G p.K256E protein...                | 291   | rs55700793   | A   | G   | NAT2   | MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQ...                        |
| NM_000015:c.C403G:p.L135V     | NM_000015       | c.C403G       | p.L135V      | 135 | L      | V      | line21115535 NM_000015 c.C403G p.L135V protein...                | 291   | rs12720065   | C   | G   | NAT2   | MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQ...                        |
| NM_000015:c.C578T:p.T193M     | NM_000015       | c.C578T       | p.T193M      | 193 | T      | M      | line21115537 NM_000015 c.C578T p.T193M protein...                | 291   | rs79050330   | C   | T   | NAT2   | MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQ...                        |
| NM_000015:c.C683T:p.P228L     | NM_000015       | c.C683T       | p.P228L      | 228 | P      | L      | line21115542 NM_000015 c.C683T p.P228L protein...                | 291   | rs45518335   | C   | T   | NAT2   | MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQ...                        |
| NM_000015:c.G191A:p.R64Q      | NM_000015       | c.G191A       | p.R64Q       | 64  | R      | Q      | line21115532 NM_000015 c.G191A p.R64Q protein-...                | 291   | rs1801279    | G   | A   | NAT2   | MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQ...                        |
| ...                           | ...             | ...           | ...          | ... | ...    | ...    | ...                                                              | ...   | ...          | ... | ... | ...    | ...                                                                      |
| NM_214711:c.C271T:p.R91C      | NM_214711       | c.C271T       | p.R91C       | 91  | R      | C      | line11007849 NM_214711 c.C271T p.R91C protein-...                | 220   | rs1613461    | C   | T   | PRR27  | MKLLLWACIVCVAFARKRRFPFIGEDDNDDGHPLHPSLNIPYGIRN...                        |
| NM_214711:c.C407A:p.A136D     | NM_214711       | c.C407A       | p.A136D      | 136 | A      | D      | line11007851 NM_214711 c.C407A p.A136D protein...                | 220   | rs187879823  | C   | A   | PRR27  | MKLLLWACIVCVAFARKRRFPFIGEDDNDDGHPLHPSLNIPYGIRN...                        |
| NM_214711:c.C524G:p.A175G     | NM_214711       | c.C524G       | p.A175G      | 175 | A      | G      | line11007854 NM_214711 c.C524G p.A175G protein...                | 220   | rs144377602  | C   | G   | PRR27  | MKLLLWACIVCVAFARKRRFPFIGEDDNDDGHPLHPSLNIPYGIRN...                        |
| NM_214711:c.G494C:p.G165A     | NM_214711       | c.G494C       | p.G165A      | 165 | G      | A      | line11007853 NM_214711 c.G494C p.G165A protein...                | 220   | rs142405912  | G   | C   | PRR27  | MKLLLWACIVCVAFARKRRFPFIGEDDNDDGHPLHPSLNIPYGIRN...                        |
| NM_214711:c.T557C:p.V186A     | NM_214711       | c.T557C       | p.V186A      | 186 | V      | A      | line11007855 NM_214711 c.T557C p.V186A protein...                | 220   | rs112603630  | T   | C   | PRR27  | MKLLLWACIVCVAFARKRRFPFIGEDDNDDGHPLHPSLNIPYGIRN...                        |

### Ld score and MAF for variants
The ld score and MAF can be obtained from GCTA.

### SNP annotation
The snp annotation can be sourced from dbSNP.
