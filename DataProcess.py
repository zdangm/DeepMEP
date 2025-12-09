from utils import *
from Protein import ProteinSeq, DiscreteEncoder, ResidueInfo
from Genome2Proteome import CommonSNP, Genome2Proteome, RareSNP


def process_uniprot_phos_data():
    # process phosphorate data from uniprot
    file_path = 'data/dataset/uniprot/uniprotkb_AND_model_organism_9606_AND_r_2024_09_02.tsv'
    public_raw_seq = pd.read_csv(file_path, sep='\t')
    public_raw_seq = public_raw_seq[~public_raw_seq['Modified residue'].isna()]
    uniprot_phos_data = []
    for i, row in public_raw_seq.iterrows():
        seq, protein, modified_type = row['Sequence'], row['Entry'], row['Modified residue']
        modified_type_split = modified_type.split('MOD_RES ')
        for split_type in modified_type_split:
            if '; /' in split_type:
                split_split_lst = split_type.split('; /')
                if (('Phosphoserine' in split_split_lst[1])
                        or ('Phosphotyrosine' in split_split_lst[1])
                        or ('Phosphothreonine' in split_split_lst[1])):
                    phos_pos_ = split_split_lst[0]
                    try:
                        pos = int(phos_pos_) - 1
                        uniprot_phos_data.append((protein, pos, seq[pos], 1))
                    except Exception as e:
                        do_nothing(e)
    uniprot_df = pd.DataFrame(uniprot_phos_data, columns=['accession', 'pos', 'aa', 'phosphorylation'])

    # check uniprot phos data
    protein_sequences = ProteinSeq().protein_sequences
    error_lst = []
    for i in range(uniprot_df.shape[0]):
        entry = uniprot_df.iloc[i].iloc[0]
        pos = uniprot_df.iloc[i].iloc[1]
        aa = uniprot_df.iloc[i].iloc[2]
        try:
            if protein_sequences[entry][pos] != aa:
                error_lst.append(entry)
        except Exception as e:
            error_lst.append(entry)
            do_nothing(e)
    uniprot_df = uniprot_df[~uniprot_df['accession'].isin(set(error_lst))]
    # uniprot_df['site_id'] = uniprot_df['accession'] + '_' + uniprot_df['pos'].astype(str) + '_' + uniprot_df['aa']
    uniprot_df.to_csv('data/dataset/uniprot/uniprot_phos_data.txt', index=False, sep='\t')
    return uniprot_df


def process_kinase_category():
    kinase_category = pd.read_csv('data/dataset/EPSD_annotation/kinase_family.csv')
    kinase_category['Name'] = kinase_category['Name'].str.upper()
    kinase_category['Group'] = kinase_category['Group'].str.upper()
    kinase_category['Family'] = kinase_category['Family'].str.upper()
    # drop kinases that do not have kinase domains
    # kinase_category = kinase_category[~(kinase_category['Kinase Domain'] == ' ')]
    return kinase_category[['Name', 'Group', 'Family', 'Kinase Domain']]


def process_epsd_kinase_phos_data():
    # https://epsd.biocuckoo.cn/Download.php
    # experiment: 'DEPOD', 'PSEA', 'Phospho.ELM', 'PhosphoSitePlus', 'PostMod', 'RegPhos'
    # predict: GPS, NetworKIN, PhosphoPICK, PKIS
    # no kinase: PhosphoNetworks; no position: HuPHO
    # step 1: load data
    phosphositeplus = pd.read_csv('data/dataset/EPSD_annotation/kinase_annotation/PhosphoSitePlus.txt', sep='\t')
    phosphositeplus['db'] = 'PhosphoSitePlus'
    phosphoELM = pd.read_csv('data/dataset/EPSD_annotation/kinase_annotation/Phospho.ELM.txt', sep='\t')
    phosphoELM['db'] = 'Phospho.ELM'
    depod = pd.read_csv('data/dataset/EPSD_annotation/kinase_annotation/DEPOD.txt', sep='\t')
    depod = depod.loc[~depod['Position'].isna(), :]
    depod.columns = ['EPSD ID', 'UniProt', 'Position', 'Kinase', 'Source']
    depod['db'] = 'DEPOD'
    regphos = pd.read_csv('data/dataset/EPSD_annotation/kinase_annotation/RegPhos.txt', sep='\t')
    regphos['db'] = 'RegPhos'
    psea = pd.read_csv('data/dataset/EPSD_annotation/kinase_annotation/PSEA.txt', sep='\t')
    psea['db'] = 'PSEA'
    postmod = pd.read_csv('data/dataset/EPSD_annotation/kinase_annotation/PostMod.txt', sep='\t')
    postmod['db'] = 'PostMod'
    kinase_anno = pd.concat([phosphositeplus, phosphoELM, depod, regphos, psea, postmod], axis=0)
    kinase_anno = kinase_anno.reset_index(drop=True)
    kinase_anno = kinase_anno.loc[~(kinase_anno['Position'] == 'N/A '), :]
    kinase_anno = kinase_anno[['UniProt', 'Position', 'Kinase', 'db']]
    # step 2: formulate data
    protein_sequences = ProteinSeq().protein_sequences
    kinase_anno = kinase_anno[kinase_anno['UniProt'].isin(protein_sequences.keys())]  # 29073
    kinase_anno = kinase_anno.reset_index(drop=True)
    pos_res = []
    for i, row in kinase_anno.iterrows():
        pos_str, entry = row['Position'], row['UniProt']
        if pos_str.isnumeric():
            pos = int(pos_str) - 1
            res = protein_sequences[entry][pos]
        else:
            res = pos_str[0]
            pos = int(pos_str[1:]) - 1
        pos_res.append((pos, res))
    pos_res = pd.DataFrame(pos_res, columns=['pos', 'aa'])
    kinase_anno = pd.concat([kinase_anno, pos_res], axis=1)
    kinase_anno = kinase_anno[kinase_anno['aa'].isin({'S', 'T', 'Y'})]  # 29072
    kinase_anno = kinase_anno[['UniProt', 'pos', 'aa', 'Kinase', 'db']]
    kinase_anno.columns = ['accession', 'pos', 'aa', 'kinase', 'db']
    kinase_anno = kinase_anno.drop_duplicates(subset=['accession', 'pos', 'aa', 'kinase'])  # 18991
    # step 3: check sequence consistent
    error_lst = []
    for i in range(kinase_anno.shape[0]):
        accession = kinase_anno.iloc[i].iloc[0]
        pos = kinase_anno.iloc[i].iloc[1]
        aa = kinase_anno.iloc[i].iloc[2]
        try:
            if protein_sequences[accession][pos] != aa:
                error_lst.append(accession)
        except Exception as e:
            error_lst.append(accession)
            do_nothing(e)
    kinase_anno = kinase_anno[~kinase_anno['accession'].isin(set(error_lst))]  # 18752
    # step 4: formulate kinase
    phos_kinase_anno = kinase_anno.copy()
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].str.replace('_group', '')
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].str.replace('(', ';', regex=False)
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].str.replace(')', '', regex=False)
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].str.replace('/', ';', regex=False)
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].str.split(';')
    phos_kinase_anno = phos_kinase_anno.explode(column='kinase')
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].str.upper()  # 20161
    replace_dict = {
        'CAMK1-ALPHA': 'CAMK1A', 'CAMK1-BETA': 'CAMK1B', 'CAMK2-ALPHA': 'CAMK2A',
        'CAMK2-BETA': 'CAMK2B', 'CAMK2-DELTA': 'CAMK2D', 'CAMK2-DELTA ISO8': 'CAMK2D',
        'CAMK2-GAMMA': 'CAMK2G', 'CDK11A': 'CDK11', 'CDK11A ISO10': 'CDK11',
        'CDK11B': 'CDK 11', 'CK1 ALPHA': 'CK1A', 'CK1 DELTA': 'CK1D',
        'CK1 EPSILON': 'CK1E', 'CK1-A': 'CK1A', 'CK1-D': 'CK1D', 'CK1-E': 'CK1E',
        'CK1-G1': 'CK1G1', 'CK1-G2': 'CK1G2', 'CK1_ALPHA': 'CK1A', 'CK1_DELTA': 'CK1D',
        'CK1_EPSILON': 'CK1E', 'CK2-A1': 'CK2A1', 'CK2-A2': 'CK2A2', 'DNA-PK': 'DNAPK',
        'FAK1': 'FAK', 'FAK2': 'FAK', 'GSK-3ALPHA': 'GSK3A', 'GSK-3BETA': 'GSK3B',
        'GSK-3_ALPHA': 'GSK3A', 'GSK-3_BETA': 'GSK3B', 'IKK-ALPHA': 'IKKA', 'IKK-BETA': 'IKKB',
        'IKK-EPSILON': 'IKKE', 'IKK_ALPHA': 'IKKA', 'IKK_BETA': 'IKKB', 'IKK_EPSILON': 'IKKE',
        'ILK1': 'ILK', 'JNK1 ISO2': 'JNK1', 'JNK2 ISO2': 'JNK2', 'LIMK1_DROME': 'LIMK1',
        'MARK3 ISO3': 'MARK3', 'MNK1 ISO2': 'MNK1', 'P38-ALPHA': 'P38A', 'P38-BETA': 'P38B',
        'P38-DELTA': 'P38D', 'P38-GAMMA': 'P38G', 'PDGFR ALPHA': 'PDGFRA', 'PDGFR BETA': 'PDGFRB',
        'PDGFR_ALPHA': 'PDGFRA', 'PDGFR_BETA': 'PDGFRB', 'PDK-1': 'PDK1', 'PKACA ISO2': 'PKACA',
        'PKC ALPHA': 'PKCA', 'PKC BETA': 'PKCB', 'PKC DELTA': 'PKCD', 'PKC EPSILON': 'PKCE',
        'PKC ETA': 'PKCE', 'PKC GAMMA': 'PKCG', 'PKC IOTA': 'PKCI', 'PKC THETA': 'PKCT',
        'PKC ZETA': 'PKCZ', 'PKCB ISO2': 'PKCB', 'PKC_ALPHA': 'PKCA', 'PKC_BETA': 'PKCB',
        'PKC_DELTA': 'PKCD', 'PKC_DROME': 'PKCD', 'PKC_EPSILON': 'PKCE', 'PKC_ETA': 'PKCE',
        'PKC_GAMMA': 'PKCG', 'PKC_IOTA': 'PKCI', 'PKC_THETA': 'PKCT', 'PKC_ZETA': 'PKCZ',
        'PKG1 ISO2': 'PKG1', 'PKG1A': 'PKG1', 'PKG1B': 'PKG1', 'RET ISO3': 'RET', 'RSK-1': 'RSK1',
        'RSK-2': 'RSK2', 'RSK-3': 'RSK3', 'SIK2': 'SIK', 'SRC ISO1': 'SRC', 'SRC-TYPE TYR-KINASES': 'SRC',
        'SRC64B_DROME': 'SRC', 'SRMS': 'SRM',
    }
    phos_kinase_anno['kinase'] = phos_kinase_anno['kinase'].replace(replace_dict)
    phos_kinase_anno = phos_kinase_anno.drop_duplicates(subset=['accession', 'pos', 'aa', 'kinase'])  # 15601
    # step 5: merge with kinase category
    kinase_category = process_kinase_category()
    merged_phos_kinase_anno = phos_kinase_anno.merge(kinase_category, left_on='kinase', right_on='Name', how='inner')
    merged_phos_kinase_anno.to_csv('data/dataset/EPSD_annotation/phos_kinase_anno.txt', sep='\t', index=False)
    # merged_phos_kinase_anno['site_id'] = (merged_phos_kinase_anno['accession'] + '_' +
    #                                       merged_phos_kinase_anno['pos'].astype(str) + '_' +
    #                                       merged_phos_kinase_anno['aa'])
    return merged_phos_kinase_anno # 11373


class ProteinSequenceAnnotation:
    def __init__(self):
        select_kinase_group = self.get_kinase_group()
        select_kinase_family = self.get_kinase_family()
        self.group2id = {select_kinase_group[i]: i for i in range(len(select_kinase_group))}
        self.id2group = {i: select_kinase_group[i] for i in range(len(select_kinase_group))}
        self.family2id = {select_kinase_family[i]: i for i in range(len(select_kinase_family))}
        self.id2family = {i: select_kinase_family[i] for i in range(len(select_kinase_family))}
        self.protein_comprehend_annotation, self.site_table, self.protein_table = self.generate_protein_comprehend_annotation()

    @staticmethod
    def get_kinase_group():
        # ['AGC', 'ATYPICAL', 'CAMK', 'CMGC', 'STE', 'TK', 'OTHER']
        return ['AGC', 'ATYPICAL', 'CAMK', 'CK1', 'CMGC', 'STE', 'TKL', 'TK', 'OTHER']

    @staticmethod
    def get_kinase_family():
        # ['AKT', 'PIKK', 'PKA', 'PKC', 'CDK', 'CK2', 'MAPK', 'SRC']
        return ['PKA', 'AKT', 'CDK', 'MAPK', 'SRC', 'CK2', 'PKC', 'PIKK']

    @staticmethod
    def get_id2gene_name() -> Dict[str, str]:
        file_path = 'data/dataset/uniprot/protein_id2gene_name.tsv'
        id2name_df = pd.read_csv(file_path, sep='\t')
        gene_names = [str(i).split(' ')[0] for i in id2name_df['Gene Names']]
        id2name_df['gene_name'] = gene_names
        protein_id2gene_name = {}
        for protein_id, gene_name in zip(id2name_df['Entry'], id2name_df['gene_name']):
            protein_id2gene_name[protein_id] = gene_name
        return protein_id2gene_name

    def generate_protein_comprehend_annotation(self) -> Tuple[Dict[str, Dict[str, Union[str, np.array]]], pd.DataFrame, pd.DataFrame]:
        """
        Generate comprehend phosphorylation annotation for every involved protein according to the database UniProt, EPSD, CBMAP.
        Note: sites without any phosphorylation annotation are classified into unphosphorylation
        return: protein_comprehend_annotation, site_table, protein_table
        :return: Tuple[[Dict[str, Dict[str, np.array]], {
            'protein_accession': {
                'protein_seq': 'AAA...'
                'is_sty': array([[0, 0, 1], [0, 0, 0], ...]),
                'uniprot_general_label': array([0, 0, 1, 0, 0, ...]),
                'lab_general_label': array([2.134754, -999, -999, -999, 1.2321, ...]),
                'epsd_kinase_group_label': array([[0, 0, 1, ..., 0, 0], ...]),
                'epsd_kinase_family_label': array([[0, 0, 1, ..., 0, 0], ...])
            }, ...], pd.DataFrame, pd.DataFrame]
        }
        """
        save_path = 'data/dataset/cache/comprehend_annotation'
        whether_load = os.path.exists(f'{save_path}/protein2full_annotation.pkl')
        whether_load &= os.path.exists(f'{save_path}/site_table.txt')
        whether_load &= os.path.exists(f'{save_path}/protein_table.txt')
        if whether_load:
            protein2full_annotation = Loader(f'{save_path}/protein2full_annotation.pkl').load_pkl()
            site_table = pd.read_csv(f'{save_path}/site_table.txt', sep='\t', low_memory=False)
            protein_table = pd.read_csv(f'{save_path}/protein_table.txt', sep='\t')
        else:
            # load uniprot, CBMAP, EPSD annotation
            uniprot_phos_data = pd.read_csv('data/dataset/uniprot/uniprot_phos_data.txt', sep='\t')
            lab_phos_data = pd.read_csv('data/dataset/CBMAP/lab_phos_data.txt', sep='\t')
            epsd_kinase_phos_data = pd.read_csv('data/dataset/EPSD_annotation/phos_kinase_anno.txt', sep='\t')
            protein_sequences: Dict = ProteinSeq().protein_sequences

            # define some necessary variables
            window = 5
            site_table = []
            site_table_columns = [
                'site_id', 'protein', 'site_res', 'lab_label', 'uniprot_label', 'group_label',
                'having_uniprot', 'having_kinase', 'having_lab', 'having_structure', 'site_fragment'
            ]
            protein_table = []
            protein_table_columns = ['protein', 'sty_count', 'having_uniprot', 'having_kinase', 'having_lab',
                                     'having_structure', 'uniprot_count', 'lab_count'] + self.get_kinase_group()
            structure_proteins = set([
                f.split('.')[0] for f in os.listdir('data/dataset/protein_res_coordinates/alphafold/protein_graph')
            ])
            focused_sites = {'S': 0, 'T': 1, 'Y': 2}
            protein2full_annotation = {}
            uniprot_proteins = set(uniprot_phos_data['accession'])
            lab_proteins = set(lab_phos_data['accession'])
            epsd_proteins = set(epsd_kinase_phos_data['accession'])
            involved_proteins = (uniprot_proteins | lab_proteins | epsd_proteins)
            protein_id2gene_name = self.get_id2gene_name()
            for protein in tqdm(involved_proteins, desc='Generating Protein Comprehend Annotation'):
                protein_seq = protein_sequences[protein]
                # initialize the full annotation
                is_sty = np.zeros((len(protein_seq), 3))
                uniprot_general_label = np.zeros(len(protein_seq))
                lab_general_label = np.zeros(len(protein_seq)) - 999
                epsd_kinase_group_label = np.zeros((len(protein_seq), len(self.group2id)))
                epsd_kinase_family_label = np.zeros((len(protein_seq), len(self.family2id)))

                # is_sty
                for pos, aa in enumerate(protein_seq):
                    if aa in {'S', 'T', 'Y'}:
                        is_sty[pos][focused_sites[aa]] = 1

                # uniprot general phosphorylation annotation
                if protein in uniprot_proteins:
                    phos_pos = uniprot_phos_data[uniprot_phos_data['accession'] == protein]['pos'].to_list()
                    uniprot_general_label[phos_pos] = 1

                # CBMAP phosphorylation intensity annotation
                if protein in lab_proteins:
                    phos_pos = lab_phos_data[lab_phos_data['accession'] == protein]['pos'].to_list()
                    mean_intensity = lab_phos_data[lab_phos_data['accession'] == protein]['standard_log_mean_intensity'].to_list()
                    lab_general_label[phos_pos] = mean_intensity

                # EPSD kinase-specific phosphorylation annotation
                if protein in epsd_proteins:
                    phos_pos = epsd_kinase_phos_data[epsd_kinase_phos_data['accession'] == protein]['pos'].to_list()
                    group_lst = epsd_kinase_phos_data[epsd_kinase_phos_data['accession'] == protein]['Group'].to_list()
                    family_lst = epsd_kinase_phos_data[epsd_kinase_phos_data['accession'] == protein]['Family'].to_list()
                    # classified by kinase group
                    for pos, group in zip(phos_pos, group_lst):
                        if group in self.group2id.keys():
                            epsd_kinase_group_label[pos][self.group2id[group]] = 1
                    # classified by kinase family
                    for pos, family in zip(phos_pos, family_lst):
                        if family in self.family2id.keys():
                            epsd_kinase_family_label[pos][self.family2id[family]] = 1

                protein2full_annotation[protein] = {
                    'protein_seq': protein_seq, 'is_sty': is_sty,
                    'uniprot_general_label': uniprot_general_label,
                    'lab_general_label': lab_general_label,
                    'epsd_kinase_group_label': epsd_kinase_group_label,
                    'epsd_kinase_family_label': epsd_kinase_family_label
                }

                # generate protein count table
                sty_count = np.sum(is_sty)
                group_count = np.zeros(len(self.group2id))
                uniprot_count, lab_count = np.sum(uniprot_general_label), np.sum(lab_general_label > -999)

                # generate site table
                sty_pos = np.where(np.sum(is_sty, axis=1) == 1)[0].tolist()
                having_uniprot = 1 if protein in uniprot_proteins else 0
                having_lab = 1 if protein in lab_proteins else 0
                having_kinase = 1 if protein in epsd_proteins else 0
                having_structure = 1 if protein in structure_proteins else 0
                padding_protein_seq = '_' * window + protein_seq + '_' * window
                for pos in sty_pos:
                    site_res = protein_seq[pos]
                    site_id = protein_id2gene_name[protein] + '_' + protein + '_' + site_res + '_' + str(pos + 1)
                    uniprot_label = uniprot_general_label[pos]
                    lab_label = lab_general_label[pos]
                    if np.sum(epsd_kinase_group_label[pos]) == 0:
                        group_label = 0
                    else:
                        group_id = np.where(epsd_kinase_group_label[pos] == 1)[0]
                        group_label = ', '.join([self.id2group[i] for i in group_id])
                        for i in group_id:
                            group_count[i] += 1
                    site_fragment = padding_protein_seq[pos: pos+2*window+1]
                    site_table.append((
                        site_id, protein, site_res, lab_label, uniprot_label, group_label,
                        having_uniprot, having_kinase, having_lab, having_structure, site_fragment
                    ))
                protein_table.append(tuple(
                    [protein, sty_count, having_uniprot, having_kinase, having_lab, having_structure,
                     uniprot_count, lab_count] + group_count.tolist()
                ))
            site_table = pd.DataFrame(site_table, columns=site_table_columns)
            # sort the df by 'general_label' to make the '1' at front
            site_table = site_table.sort_values(by=[
                'lab_label', 'having_lab', 'uniprot_label', 'having_uniprot', 'group_label',
                'having_kinase', 'having_structure'
            ], ascending=False).reset_index(drop=True) # 1189688
            # filter same instances (the positive in same instances would be kept)
            site_table = site_table.drop_duplicates(subset=['site_fragment']).reset_index(drop=True) # 1146170
            protein_table = pd.DataFrame(protein_table, columns=protein_table_columns)
            Saver(protein2full_annotation, f'{save_path}/protein2full_annotation.pkl').save_pkl()
            site_table.to_csv(f'{save_path}/site_table.txt', sep='\t', index=False)
            protein_table.to_csv(f'{save_path}/protein_table.txt', sep='\t', index=False)
        return protein2full_annotation, site_table, protein_table

    def split_dataset_at_protein_level(self, clean=False, mode=None):
        save_path = f'data/dataset/cache/train_val_test_proteins.pkl'
        if clean:
            os.system(f'rm {save_path}')
        if os.path.exists(save_path):
            return Loader(save_path).load_pkl()
        else:
            all_proteins = set(self.protein_table['protein'])
            lab_proteins = set(self.protein_table[self.protein_table['having_lab'] == 1]['protein'])
            uniprot_proteins = set(self.protein_table[self.protein_table['having_uniprot'] == 1]['protein'])
            epsd_proteins = set(self.protein_table[self.protein_table['having_kinase'] == 1]['protein'])
            structure_proteins = set(self.protein_table[self.protein_table['having_structure'] == 1]['protein'])
            all_involved_proteins = lab_proteins & uniprot_proteins & epsd_proteins & structure_proteins
            other_lab_proteins = lab_proteins - all_involved_proteins
            other_proteins = list(all_proteins - all_involved_proteins - other_lab_proteins)
            all_involved_proteins = list(all_involved_proteins)
            other_lab_proteins = list(other_lab_proteins)
            random.seed(1009)
            random.shuffle(all_involved_proteins)
            random.shuffle(other_lab_proteins)
            random.shuffle(other_proteins)

            def split_lst(lst):
                l1, l2 = int(len(lst) * 0.7), int(len(lst) * 0.85)
                return lst[:l1], lst[l1:l2], lst[l2:]

            train1, val1, test1 = split_lst(all_involved_proteins)
            train2, val2, test2 = split_lst(other_lab_proteins)
            train3, val3, test3 = split_lst(other_proteins)
            train_proteins = set(train1 + train2 + train3)
            val_proteins = set(val1 + val2 + val3)
            test_proteins = set(test1 + test2 + test3)
            Saver((train_proteins, val_proteins, test_proteins), save_path).save_pkl()
            return train_proteins, val_proteins, test_proteins


class ProteinStructureFeature:
    def __init__(self):
        self.structure_dir = 'data/dataset/protein_res_coordinates/alphafold/protein_graph'
        self.Residue = ResidueInfo()

    def compute_all_sites_structure(self):
        site_table = ProteinSequenceAnnotation().site_table
        site_ids = set(site_table[site_table['having_structure'] == 1]['site_id'])
        for site_id in tqdm(site_ids, desc='Computing all sites structure features'):
            site_split = site_id.split('_')
            protein = f"{site_split[0]}_{site_split[1]}"
            pos = int(site_split[-1])
            structure_feature = self.select_protein_structure(protein_name=protein, pos=pos)
            Saver(structure_feature, filename=f'data/dataset/structure_features/{protein}_{pos}.pkl').save_pkl()

    def selectProteinStructure(self, protein_name: str, pos: int):
        file_name = f'data/dataset/structure_features/{protein_name}_{pos}.pkl'
        if os.path.exists(file_name):
            return Loader(file_name).load_pkl()
        else:
            structure_feature = self.select_protein_structure(protein_name=protein_name, pos=pos)
            Saver(structure_feature, filename=f'data/dataset/structure_features/{protein_name}_{pos}.pkl').save_pkl()
            return structure_feature

    def select_protein_structure(self, protein_name: str, pos: int, site_num: int = 30,
                                 structure_feature_nums: int = 44) -> np.array:
        # return: (SecondStructure, PSI_PHI, RSA, Sequence)
        file_path = self.structure_dir + '/' + protein_name + '.pkl'
        proteinStructure = Loader(file_path).load_pkl()
        site_dist = proteinStructure['DistMatrix'][pos]
        site_index = np.argsort(site_dist)[:site_num]  # choose 30 closed sites around 'S'
        site_SS = proteinStructure['SecondStructure'][site_index]  # sites, 8
        site_PHIPSI = proteinStructure['PSI_PHI'][site_index]  # sites, 4
        site_RSA = proteinStructure['RSA'][site_index]  # sites, 5
        site_sites = [proteinStructure['Sequence'][i] for i in site_index]
        # add physical properties
        physical_properties = []
        for site in site_sites:
            fea = [
                      self.Residue.getAtomNum()[site],  # 1
                      self.Residue.getChainChargeNum()[site],  # 1
                      self.Residue.getSideChainHydrogenBondNum()[site],  # 1
                      self.Residue.getSideChainPKa()[site],  # 1
                      self.Residue.getHydrophobicity()[site]  # 1
                  ] + self.Residue.getOneHotDic()[site]  # 22
            physical_properties.append(fea)
        physical_properties = np.array(physical_properties)  # sites, 5+22
        structure_features = np.concatenate([
            site_SS, site_PHIPSI, site_RSA, physical_properties
        ], axis=1)  # sites, 44
        # impute lacked sites with 0
        if len(site_index) < site_num:
            pad_site = [0] * structure_feature_nums
            padding_sites = np.array([pad_site for _ in range(site_num - len(site_index))])
            structure_features = np.concatenate([structure_features, padding_sites], axis=0)
        return structure_features


class TrainingDataset:
    def __init__(self,
                 window_size: int, # the scale size of context
                 use_structure: bool, # whether using structure data
                 data_from: Literal['uniprot', 'epsd_group', 'epsd_family', 'cbmap'], # the source of data
                 ):
        self.window_size = window_size
        self.protein_seq_anno = ProteinSequenceAnnotation()
        self.protein_structure = ProteinStructureFeature()
        self.data_from = data_from
        self.use_structure = use_structure
        self.focus_site2vec = {'S': 0, 'T': 1, 'Y': 2}
        site_table = self.protein_seq_anno.site_table.copy()
        if data_from == 'uniprot':
            site_table = site_table[site_table['having_uniprot'] == 1]
        elif data_from == 'epsd_group' or data_from == 'epsd_family':
            site_table = site_table[site_table['having_kinase'] == 1]
        elif data_from == 'cbmap':
            site_table = site_table[site_table['lab_label'] > -999]
        else:
            site_table = site_table
        if use_structure:
            site_table = site_table[site_table['having_structure'] == 1]
        self.site_table = site_table

    def getSeqInstance(self, clean=False) -> Tuple:
        temp_dir = (f'data/dataset/cache/train_seqs/processed_raw_seq_'
                    f'{self.window_size}_{self.data_from}_{self.use_structure}')
        whether_load = self.use_structure
        if self.use_structure:
            if clean:
                os.system(f'rm -r {temp_dir}')
            make_dir(temp_dir)
            whether_load = os.path.exists(temp_dir + '/label.npy')
            whether_load &= os.path.exists(temp_dir + '/context.pkl')
            whether_load &= os.path.exists(temp_dir + '/mask_instance.npy')
            whether_load &= os.path.exists(temp_dir + '/site_id_lst.pkl')
            whether_load &= os.path.exists(temp_dir + '/structure.npy')
        if whether_load:
            print('Using cache data to return Seq Instance! ')
            label_instance = Loader(temp_dir + '/label.npy').load_npy()
            context_instance = Loader(temp_dir + '/context.pkl').load_pkl()
            site_id_lst = Loader(temp_dir + '/site_id_lst.pkl').load_pkl()
            structure_instance = Loader(temp_dir + '/structure.npy').load_npy()
            mask_instance = Loader(temp_dir + '/mask_instance.npy').load_npy()
            return label_instance, context_instance, structure_instance, mask_instance, site_id_lst
        else:
            print('Start generating seq instances!')
            site_id_lst, label_instance, context_instance, mask_instance, structure_instance = [], [], [], [], []
            for i_site_id, i_label, i_seq, i_seq_mask, i_structure_features in self._getSingleSeqInstance_():
                if self.data_from in {'uniprot', 'cbmap'}:
                    label_instance.append(i_label.reshape(1, -1))
                else:
                    label_instance.append(i_label.reshape(1, 2 * self.window_size + 1, -1))
                context_instance.append(i_seq)
                site_id_lst.append(i_site_id)
                mask_instance.append(i_seq_mask)
                structure_instance.append(i_structure_features)
            label_instance = np.concatenate(label_instance, axis=0)
            structure_instance = np.array(structure_instance)
            mask_instance = np.array(mask_instance)
            # expand y
            if self.data_from == 'uniprot':
                label_instance_reverse = (label_instance + 1) % 2
                label_instance = np.stack([label_instance_reverse, label_instance], axis=-1)
            if self.data_from == 'epsd_group' or self.data_from == 'epsd_family':
                label_instance_reverse = (np.sum(label_instance, axis=-1) == 0).astype(int)
                label_instance_reverse = label_instance_reverse.reshape(label_instance.shape[0], -1, 1)
                label_instance = np.concatenate([label_instance_reverse, label_instance], axis=-1)
            if self.use_structure:
                make_dir(temp_dir)
                Saver(context_instance, temp_dir + '/context.pkl').save_pkl()
                Saver(label_instance, temp_dir + '/label.npy').save_npy()
                Saver(site_id_lst, temp_dir + '/site_id_lst.pkl').save_pkl()
                Saver(structure_instance, temp_dir + '/structure.npy').save_npy()
                Saver(mask_instance, temp_dir + '/mask_instance.npy').save_npy()
            return label_instance, context_instance, structure_instance, mask_instance, site_id_lst

    def _getSingleSeqInstance_(self, padding: str = '_') -> List[Tuple]:
        # return: [(site_id, label, context, structure), ...]
        # context: '___S____'; structure: np.array, (site, 44)
        # get train seq, only sliced sequences with protein full sequences will be kept
        seq_instance = []
        legal_site_ids = set(self.site_table['site_id'])
        co_proteins = set(self.site_table['protein'])
        protein_id2gene_name = self.protein_seq_anno.get_id2gene_name()
        for protein_id in tqdm(co_proteins, desc='Seq Instances'):
            protein_comprehend_anno = self.protein_seq_anno.protein_comprehend_annotation[protein_id]
            protein_seq = protein_comprehend_anno['protein_seq']
            full_seq = padding * self.window_size + protein_seq + padding * self.window_size
            # focus_pos = [i + self.window_size for i in range(len(protein_seq)) if protein_seq[i] in focus]
            full_seq_mask = protein_comprehend_anno['is_sty']
            focus_pos = np.where(np.sum(full_seq_mask, axis=1))[0] + self.window_size
            full_seq_mask = np.concatenate([np.zeros((self.window_size, 3)), full_seq_mask,
                                            np.zeros((self.window_size, 3))], axis=0)
            if self.data_from == 'uniprot':
                full_label = protein_comprehend_anno['uniprot_general_label']
                full_label = np.concatenate([
                    np.zeros(self.window_size), full_label, np.zeros(self.window_size)
                ])
            elif self.data_from == 'cbmap':
                full_label = protein_comprehend_anno['lab_general_label']
                full_label = np.concatenate([
                    np.zeros(self.window_size), full_label, np.zeros(self.window_size)
                ])
            elif self.data_from == 'epsd_group':
                padding_length = len(self.protein_seq_anno.group2id)
                kinase_label_name = 'epsd_kinase_group_label'
                full_label = protein_comprehend_anno[kinase_label_name]
                full_label = np.concatenate([
                    np.zeros((self.window_size, padding_length)), full_label,
                    np.zeros((self.window_size, padding_length))
                ])
            elif self.data_from == 'epsd_family':
                padding_length = len(self.protein_seq_anno.family2id)
                kinase_label_name = 'epsd_kinase_family_label'
                full_label = protein_comprehend_anno[kinase_label_name]
                full_label = np.concatenate([
                    np.zeros((self.window_size, padding_length)), full_label,
                    np.zeros((self.window_size, padding_length))
                ])
            else:
                full_label = None

            for pos in focus_pos:
                label = full_label[pos - self.window_size: pos + self.window_size + 1]
                seq_context = full_seq[pos - self.window_size: pos + self.window_size + 1]
                seq_mask = full_seq_mask[pos - self.window_size: pos + self.window_size + 1]
                site_id = f"{protein_id2gene_name[protein_id]}_{protein_id}_{full_seq[pos]}_{pos-self.window_size+1}"
                if site_id in legal_site_ids:
                    if self.use_structure:
                        structure_features = self.protein_structure.selectProteinStructure(
                            protein_name=protein_id, pos=pos - self.window_size
                        )
                    else:
                        structure_features = None
                    seq_instance.append((site_id, label, seq_context, seq_mask, structure_features))
        return seq_instance

    def get_train_val_test_sites(self, site_ids: Set[str]) -> Tuple:
        # this method split train-val-test
        # return train-val-test indices according to the provided site_ids
        # step 1: generate gold split reference
        train_proteins, val_proteins, test_proteins = self.protein_seq_anno.split_dataset_at_protein_level()
        site_table = self.site_table
        train_set_ids = set(site_table.loc[site_table['protein'].isin(train_proteins), 'site_id'])
        val_set_ids = set(site_table.loc[site_table['protein'].isin(val_proteins), 'site_id'])
        test_set_ids = set(site_table.loc[site_table['protein'].isin(test_proteins), 'site_id'])
        # step 2: return provided site_ids' train-val-test indices according to the gold reference
        site_id_df = pd.DataFrame({'site_id': site_ids})
        train_indices = site_id_df[site_id_df['site_id'].isin(train_set_ids)].index.to_list()
        test_indices = site_id_df[site_id_df['site_id'].isin(test_set_ids)].index.to_list()
        val_indices = site_id_df[site_id_df['site_id'].isin(val_set_ids)].index.to_list()
        return train_indices, test_indices, val_indices

    def get_set_size(self, label_instance, site_id_lst, whether_keep_test):
        train_indices, test_indices, val_indices = self.get_train_val_test_sites(site_ids=site_id_lst)
        test_indices = [test_indices[i] for i, b in enumerate(whether_keep_test) if b]

        def _print_size_(__get_info__, train_, test_, val_, total_):
            df = {'Train-Set': __get_info__(label=train_), 'Test-Set': __get_info__(label=test_),
                  'Val-Set': __get_info__(label=val_), 'Total': __get_info__(label=total_)}
            print(pd.DataFrame(df).T)

        if self.data_from == 'uniprot':
            train_label = label_instance[train_indices, self.window_size, 1]
            val_label = label_instance[val_indices, self.window_size, 1]
            test_label = label_instance[test_indices, self.window_size, 1]
            total_label = label_instance[:, self.window_size, 1]

            def _get_info_(label):
                p, t = sum(label), len(label)
                print_str = {
                    'Positive': p, 'Negative': t - p, 'Total': t, 'Ratio': round((t - p) / p, 2)
                }
                return print_str

            _print_size_(_get_info_, train_label, test_label, val_label, total_label)

        elif self.data_from == 'epsd_group' or self.data_from == 'epsd_family':
            train_label = label_instance[train_indices, self.window_size, :]
            val_label = label_instance[val_indices, self.window_size, :]
            test_label = label_instance[test_indices, self.window_size, :]
            total_label = label_instance[:, self.window_size, :]

            def _get_info_(label):
                print_str = {}
                group_dict = self.protein_seq_anno.group2id if self.data_from == 'epsd_group' else self.protein_seq_anno.family2id
                for group, i in group_dict.items():
                    print_str[group] = sum(label[:, i + 1])
                print_str['Negative'] = np.sum(label[:, 0])
                print_str['Positive'] = label.shape[0] - print_str['Negative']
                print_str['Total'] = label.shape[0]
                print_str['Ratio'] = round(print_str['Negative'] / print_str['Positive'], 2)
                return print_str

            _print_size_(_get_info_, train_label, test_label, val_label, total_label)

        elif self.data_from == 'cbmap':
            train_label = label_instance[train_indices, self.window_size]
            val_label = label_instance[val_indices, self.window_size]
            test_label = label_instance[test_indices, self.window_size]
            total_label = label_instance[:, self.window_size]

            def _get_info_(label):
                return {
                    'Num': label.shape[0], 'Max': np.max(label), 'Median': np.median(label),
                    'Min': np.min(label), 'Mean': np.mean(label)
                }

            _print_size_(_get_info_, train_label, test_label, val_label, total_label)

    def getDataset(self,
                   encode_mode: Literal['one-hot', 'esm-embedding'], # the coding method for protein sequences
                   select_res: Literal['S', 'T', 'Y'] = None,
                   require_train: bool = True, require_test: bool = True, require_val: bool = True,
                   para_i: int = None) -> Dict:
        dataset = {}
        encoder = DiscreteEncoder(add_pos=False)
        label_instance, context_instance, structure_instance, mask_instance, site_id_lst = self.getSeqInstance()
        if select_res is None:
            mask_instance = np.sum(mask_instance, axis=-1)
        else:
            res_i = self.focus_site2vec[select_res]
            select_indices = mask_instance[:, self.window_size, res_i] == 1
            label_instance = label_instance[select_indices]
            structure_instance = structure_instance[select_indices]
            mask_instance = mask_instance[select_indices, :, res_i]
            id_indicies = np.arange(0, select_indices.shape[0])[select_indices]
            context_instance = [context_instance[i] for i in id_indicies]
            site_id_lst = [site_id_lst[i] for i in id_indicies]

        def _split_instances_(indices):
            context_lst = [(label_instance[i], context_instance[i]) for i in indices]
            seq_mask = torch.tensor(np.array([mask_instance[i] for i in indices])).float()
            if self.use_structure:
                structures = torch.tensor([structure_instance[i] for i in indices]).float().transpose(1, 2)
            else:
                structures = None
            site_ids = [site_id_lst[i] for i in indices]
            return context_lst, seq_mask, structures, site_ids

        train_indices, test_indices, val_indices = self.get_train_val_test_sites(site_ids=site_id_lst)
        context_train_lst, train_seq_mask, structure_train, train_site_ids = _split_instances_(train_indices)
        context_test_lst, test_seq_mask, structure_test, test_site_ids = _split_instances_(test_indices)
        context_val_lst, val_seq_mask, structure_val, val_site_ids = _split_instances_(val_indices)

        # remove instances in test-set that are highly similar with train-set
        def _hamming_similarity_(s1, s2):
            distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))
            return 1 - distance / len(s1)

        cache_path = f'data/dataset/cache/keep_test_set_{self.data_from}_{self.use_structure}.pkl'
        if os.path.exists(cache_path):
            whether_keep = Loader(cache_path).load_pkl()
        else:
            test_set_len = len(context_test_lst)
            print(test_set_len)
            i_lst = list(range(0, test_set_len, 2000))
            if para_i is not None:
                if para_i == len(i_lst)-1:
                    start_, end_ = i_lst[para_i], test_set_len-1
                else:
                    start_, end_ = i_lst[para_i], i_lst[para_i+1]
            else:
                start_, end_ = 0, test_set_len-1
            whether_keep, threshold = [True] * test_set_len, 0.8
            # step 1: keep test-set independence
            for i in tqdm(range(start_, end_), desc='Step 1. Keeping dependency...'):
                for j in range(i + 1, test_set_len):
                    hs = _hamming_similarity_(context_test_lst[i][1], context_test_lst[j][1])
                    if hs >= threshold:
                        whether_keep[i] = False
                        break
            # step 2: remove redundant instances in test-set
            for i in tqdm(range(start_, end_), desc='Step 2. Removing similar instances with train-set...'):
                if whether_keep[i]:
                    for j in range(len(context_train_lst)):
                        hs = _hamming_similarity_(context_test_lst[i][1], context_train_lst[j][1])
                        if hs >= threshold:
                            whether_keep[i] = False
                            break
            Saver(whether_keep, f'data/dataset/cache/parallel/{para_i}_'
                                f'{self.data_from}_{self.use_structure}.pkl').save_pkl()
        self.get_set_size(label_instance, site_id_lst, whether_keep)
        print(f'Removing {np.sum(np.array(whether_keep) == False)} instances in test-set!')

        context_test_lst = [context_test_lst[i] for i in range(len(context_test_lst)) if whether_keep[i]]
        test_seq_mask = test_seq_mask[whether_keep]
        if self.use_structure:
            structure_test = structure_test[whether_keep]
        test_site_ids = [test_site_ids[i] for i in range(len(test_site_ids)) if whether_keep[i]]

        dataset['structure'] = [structure_train, structure_test, structure_val]
        dataset['site_id'] = train_site_ids, test_site_ids, val_site_ids

        if encode_mode == 'one-hot':
            train_seq_label, train_seq_vec = None, None
            test_seq_label, test_seq_vec = None, None
            val_seq_label, val_seq_vec = None, None
            if require_train:
                train_seq_label, train_seq_vec = encoder.one_hot(context_train_lst)
                train_seq_label = torch.tensor(train_seq_label).float()
                train_seq_vec = torch.tensor(train_seq_vec).transpose(1, 2).float()
            if require_test:
                test_seq_label, test_seq_vec = encoder.one_hot(context_test_lst)
                test_seq_label = torch.tensor(test_seq_label).float()
                test_seq_vec = torch.tensor(test_seq_vec).transpose(1, 2).float()
            if require_val:
                val_seq_label, val_seq_vec = encoder.one_hot(context_val_lst)
                val_seq_label = torch.tensor(val_seq_label).float()
                val_seq_vec = torch.tensor(val_seq_vec).transpose(1, 2).float()
            dataset['context'] = [train_seq_label, train_seq_vec, train_seq_mask,
                                  test_seq_label, test_seq_vec, test_seq_mask,
                                  val_seq_label, val_seq_vec, val_seq_mask]

        if encode_mode == 'esm-embedding':
            train_seq_label, train_seq_str, test_seq_label, test_seq_str = [], [], [], []
            val_seq_label, val_seq_str = [], []
            if require_train:
                for label_i, seq_i in context_train_lst:
                    train_seq_label.append(label_i)
                    train_seq_str.append(seq_i)
                train_seq_label = torch.tensor(np.array(train_seq_label)).float()
            if require_test:
                for label_i, seq_i in context_test_lst:
                    test_seq_label.append(label_i)
                    test_seq_str.append(seq_i)
                test_seq_label = torch.tensor(np.array(test_seq_label)).float()
            if require_val:
                for label_i, seq_i in context_val_lst:
                    val_seq_label.append(label_i)
                    val_seq_str.append(seq_i)
                val_seq_label = torch.tensor(np.array(val_seq_label)).float()
            dataset['context'] = [train_seq_label, train_seq_str, train_seq_mask,
                                  test_seq_label, test_seq_str, test_seq_mask,
                                  val_seq_label, val_seq_str, val_seq_mask]
        return dataset


class CrossValidationDataset:
    def __init__(self,
                 window_size: int,  # the scale size of context
                 use_structure: bool,  # whether using structure data
                 data_from: Literal['uniprot', 'epsd_group', 'epsd_family', 'cbmap'],  # the source of data
                 k_folds: int = 5,
                 ):
        self.datasetor = TrainingDataset(window_size, use_structure, data_from)
        self.k_folds = k_folds

    def get_cv_dataset(self, fold_i, encode_mode='esm-embedding'):
        dataset = {}
        label_instance, context_instance, structure_instance, mask_instance, site_id_lst = self.datasetor.getSeqInstance()
        mask_instance = np.sum(mask_instance, axis=-1)
        cv_proteins = self.get_cv_proteins()[f'fold_{fold_i}']
        # get train-test indices
        train_proteins, test_proteins = cv_proteins['train-set'], cv_proteins['test-set']
        site_table = self.datasetor.protein_seq_anno.site_table
        train_set_ids = set(site_table.loc[site_table['protein'].isin(train_proteins), 'site_id'])
        test_set_ids = set(site_table.loc[site_table['protein'].isin(test_proteins), 'site_id'])
        site_id_df = pd.DataFrame({'site_id': site_id_lst})
        train_indices = site_id_df[site_id_df['site_id'].isin(train_set_ids)].index.to_list()
        test_indices = site_id_df[site_id_df['site_id'].isin(test_set_ids)].index.to_list()

        def _split_instances_(indices):
            context_lst = [(label_instance[ii], context_instance[ii]) for ii in indices]
            seq_mask = torch.tensor(np.array([mask_instance[ii] for ii in indices])).float()
            structures = None
            site_ids = [site_id_lst[ii] for ii in indices]
            return context_lst, seq_mask, structures, site_ids

        context_train_lst, train_seq_mask, structure_train, train_site_ids = _split_instances_(train_indices)
        context_test_lst, test_seq_mask, structure_test, test_site_ids = _split_instances_(test_indices)
        dataset['structure'] = [structure_train, structure_test]
        dataset['site_id'] = train_site_ids, test_site_ids

        if encode_mode == 'one-hot':
            encoder = DiscreteEncoder(add_pos=False)
            train_seq_label, train_seq_vec = encoder.one_hot(context_train_lst)
            train_seq_label = torch.tensor(train_seq_label).float()
            train_seq_vec = torch.tensor(train_seq_vec).transpose(1, 2).float()
            test_seq_label, test_seq_vec = encoder.one_hot(context_test_lst)
            test_seq_label = torch.tensor(test_seq_label).float()
            test_seq_vec = torch.tensor(test_seq_vec).transpose(1, 2).float()
            dataset['context'] = [train_seq_label, train_seq_vec, train_seq_mask,
                                  None, None, None,
                                  test_seq_label, test_seq_vec, test_seq_mask]
        else:
            train_seq_label, train_seq_str, test_seq_label, test_seq_str = [], [], [], []
            for label_i, seq_i in context_train_lst:
                train_seq_label.append(label_i)
                train_seq_str.append(seq_i)
            train_seq_label = torch.tensor(np.array(train_seq_label)).float()
            for label_i, seq_i in context_test_lst:
                test_seq_label.append(label_i)
                test_seq_str.append(seq_i)
            test_seq_label = torch.tensor(np.array(test_seq_label)).float()

            dataset['context'] = [
                train_seq_label, train_seq_str, train_seq_mask,
                None, None, None,
                test_seq_label, test_seq_str, test_seq_mask
            ]
        return dataset

    def get_cv_proteins(self):
        cache_path = 'data/dataset/cache/cv_train_test_proteins.pkl'
        if os.path.exists(cache_path):
            return Loader(cache_path).load_pkl()
        else:
            protein_table = self.datasetor.protein_seq_anno.protein_table
            all_proteins = set(protein_table['protein'])
            lab_proteins = set(protein_table[protein_table['having_lab'] == 1]['protein'])
            uniprot_proteins = set(protein_table[protein_table['having_uniprot'] == 1]['protein'])
            epsd_proteins = set(protein_table[protein_table['having_kinase'] == 1]['protein'])
            all_involved_proteins = lab_proteins & uniprot_proteins & epsd_proteins
            other_lab_proteins = lab_proteins - all_involved_proteins
            other_proteins = list(all_proteins - all_involved_proteins - other_lab_proteins)
            all_involved_proteins = list(all_involved_proteins)
            other_lab_proteins = list(other_lab_proteins)

            random.seed(1009)
            random.shuffle(all_involved_proteins)
            random.shuffle(other_lab_proteins)
            random.shuffle(other_proteins)

            def _split_lst_(lst):
                split_lst = []
                step = math.ceil(len(lst) / self.k_folds)
                for ii in range(self.k_folds):
                    split_lst.append(lst[ii * step:(ii + 1) * step])
                return split_lst

            all_involved_proteins_lst = _split_lst_(all_involved_proteins)
            other_lab_proteins_lst = _split_lst_(other_lab_proteins)
            other_proteins_lst = _split_lst_(other_proteins)
            all_proteins_lst = [
                all_involved_proteins_lst[i] + other_lab_proteins_lst[i] + other_proteins_lst[i]
                for i in range(self.k_folds)
            ]
            cv_proteins = {}
            for i in range(self.k_folds):
                cv_proteins[f'fold_{i}'] = {
                    'test-set': set(all_proteins_lst[i]),
                    'train-set': all_proteins - set(all_proteins_lst[i])
                }
            Saver(cv_proteins, cache_path).save_pkl()
            return cv_proteins


class ApplicationDataset:
    def __init__(self):
        self.padding = '_'
        self.seq_data = self.read_missense_annotation()
        self.focused_sites = {'S', 'T', 'Y'}

    @staticmethod
    def read_missense_annotation() -> pd.DataFrame:
        data_path = 'data/genome/missense/annotation_GRch38_common_all_20180418_annovar_output.missense_SNP_with_protein_AA.tsv'
        seq_data = pd.read_csv(data_path, sep='\t')
        selected_cols = ['SNP', 'aa_ref', 'aa_alt', 'ref', 'alt', 'gene', 'sequence', 'pos']
        seq_data = seq_data[selected_cols].rename(columns={'ref': 'base_ref', 'alt': 'base_alt'})
        seq_data['i_name'] = (seq_data['SNP'].astype(str) + '_' + seq_data['pos'].astype(str) + '_' +
                              seq_data['aa_ref'].astype(str) + '_' + seq_data['aa_alt'].astype(str))
        seq_data = seq_data[['SNP', 'aa_ref', 'aa_alt', 'base_ref', 'gene', 'sequence', 'pos', 'i_name']].drop_duplicates().merge(
            seq_data.groupby('i_name')['base_alt'].agg(', '.join).reset_index(), on='i_name'
        )
        # independent_snps = set(seq_data['SNP'].drop_duplicates(keep=False))
        # seq_data = seq_data.loc[seq_data['SNP'].isin(independent_snps), :]
        return seq_data

    def getApplicationInstances(self, window_size: int) -> pd.DataFrame:
        # return: [(rsid, protein_name, ref_input_seq, alt_input_seq, distance), ..., ()]
        # drop snps having multiple variants
        seq_data = self.seq_data.copy()
        application_instances = []
        for i in tqdm(range(seq_data.shape[0]), desc='Get Application Instances By Transcript'):
            snp_row = seq_data.iloc[i]
            __padding = self.padding * window_size * 2
            full_seq = __padding + snp_row.loc['sequence'].strip('*') + __padding
            variant_pos = snp_row.loc['pos'] - 1 + window_size * 2
            aa_ref, aa_alt = snp_row.loc['aa_ref'], snp_row.loc['aa_alt']
            processed_alt_seq = full_seq[variant_pos - window_size * 2: variant_pos + window_size * 2 + 1]
            processed_ref_seq = alter_string_at_pos(processed_alt_seq, window_size * 2, aa_ref)
            left, right = window_size, len(processed_ref_seq) - window_size - 1
            ref_sty_pos = set(i for i in self.find_STY_pos(processed_ref_seq) if (i >= left) and (i <= right))
            alt_sty_pos = set(i for i in self.find_STY_pos(processed_alt_seq) if (i >= left) and (i <= right))
            ref_split_seq, alt_split_seq, variant_sty_distance, sty_changed_lst = [], [], [], []
            # normal mutation
            for pos in ref_sty_pos & alt_sty_pos:
                ref_split_seq.append(processed_ref_seq[pos - window_size:pos + window_size + 1])
                alt_split_seq.append(processed_alt_seq[pos - window_size:pos + window_size + 1])
                variant_sty_distance.append(pos - window_size * 2)
                sty_changed_lst.append('unchange')
            # STY -> aa
            for pos in ref_sty_pos - alt_sty_pos:
                ref_split_seq.append(processed_ref_seq[pos - window_size:pos + window_size + 1])
                alt_split_seq.append('')
                variant_sty_distance.append(pos - window_size * 2)
                sty_changed_lst.append('sty2aa')
            # aa -> STY
            for pos in alt_sty_pos - ref_sty_pos:
                ref_split_seq.append('')
                alt_split_seq.append(processed_alt_seq[pos - window_size:pos + window_size + 1])
                variant_sty_distance.append(pos - window_size * 2)
                sty_changed_lst.append('aa2sty')

            application_instances += list(zip(
                [snp_row.loc['i_name']] * len(ref_split_seq),
                ref_split_seq, alt_split_seq, variant_sty_distance, sty_changed_lst
            ))
        application_instances = pd.DataFrame(
            application_instances, columns=['i_name', 'ref_seq', 'alt_seq', 'variant_sty_distance', 'sty_changed']
        )

        return application_instances

    def getSTYSnps(self) -> Tuple[Set, Set]:
        # get SNPs that change the AA to 'STY'
        # return: {'rsid', 'rsid', ...}
        sty2aa_snps = set(self.seq_data.loc[self.seq_data['aa_ref'].isin(self.focused_sites), 'SNP'])
        aa2sty_snps = set(self.seq_data.loc[self.seq_data['aa_alt'].isin(self.focused_sites), 'SNP'])
        return sty2aa_snps, aa2sty_snps

    def find_STY_pos(self, seq: str) -> List:
        return [i for i, aa in enumerate(seq) if aa in self.focused_sites]


class GWASProcessor:
    def __init__(self):
        self.p_threshold = 5e-8
        self.gwas_rs_path = 'data/genome/enrichment/gwas_sig_all_snps.pkl'
        if os.path.exists(self.gwas_rs_path):
            self.gwas_rs = Loader(self.gwas_rs_path).load_pkl()
        else:
            self.gwas_rs = {'All_disease': set()}

    def count_sig_num(self):
        return {d: len(self.gwas_rs[d]) for d in self.get_diseases()}

    def get_diseases(self) -> List:
        diseases = [i for i in self.gwas_rs.keys() if i != 'All_disease']
        diseases.sort(key=lambda x: x[0])
        diseases.append('All_disease')
        return diseases

    def process_NEW202_GWAS(self):
        # return: Dict['disease': [rsid, rsid, ...], ...]
        file_dir = '/data/shared_data/neuropsych_GWAS/EUR/processed'
        file_names = [
            '/ALS/ALS_processed.txt', '/ASD/ASD_processed.txt', '/BIP/BIP_processed.txt',
            '/Insomnia/Insomnia_processed.txt', '/TAAU/EUR_stratified/AgeSmk_processed.txt',
            '/TAAU/EUR_stratified/CigDay_processed.txt', '/TAAU/EUR_stratified/DrnkWk_processed.txt',
            '/TAAU/EUR_stratified/SmkCes_processed.txt', '/TAAU/EUR_stratified/SmkInit_processed.txt',
            '/BMI/BMI_processed.txt', '/AD/file2/ADfile2_processed.txt', '/SCZ/file2/SCZfile2_processed.txt',
            '/PD/PD_processed.txt', '/IQ/IQ_processed.txt', '/PTSD/PTSD_processed.txt', '/MDD/MDD_processed.txt',
            '/ADHD/ADHD_processed.txt', '/ANX/ANX_processed.txt', '/ED/ED_processed.txt', '/EA/EA_processed.txt',
        ]
        diseases = [
            'ALS', 'ASD', 'BIP', 'Insomnia', 'TAAU-AgeSmk', 'TAAU-CigDay', 'TAAU-DrnkWk', 'TAAU-SmkCes',
            'TAAU-SmkInit', 'BMI', 'AD', 'SCZ', 'PD', 'IQ', 'PTSD', 'MDD', 'ADHD', 'ANX', 'ED', 'EA',
        ]
        all_disease_snps = self.gwas_rs['All_disease']
        for i, file_name in tqdm(enumerate(file_names), desc='Processing GWAS...'):
            if diseases[i] not in self.gwas_rs.keys():
                temp_rs = pd.read_csv(file_dir + file_name, sep='\t')
                self.gwas_rs[diseases[i]] = set(temp_rs.loc[temp_rs['p'] <= self.p_threshold, 'rsid'])
                all_disease_snps.update(self.gwas_rs[diseases[i]])
        self.gwas_rs['All_disease'] = all_disease_snps
        Saver(self.gwas_rs, self.gwas_rs_path).save_pkl()


class RAVARProcessor:
    def __init__(self):
        self.ravar_rs_path = 'data/genome/enrichment/ravar_sig_all_snps.pkl'
        if os.path.exists(self.ravar_rs_path):
            self.disease2rare_variants = Loader(self.ravar_rs_path).load_pkl()
        else:
            self.disease2rare_variants = {'All_disease': set()}

    def count_sig_num(self):
        return {d: len(self.disease2rare_variants[d]) for d in self.get_diseases()}

    def get_diseases(self) -> List:
        diseases = [i for i in self.disease2rare_variants.keys() if i != 'All_disease']
        diseases.sort(key=lambda x: x[0])
        diseases.append('All_disease')
        return diseases

    def generate_rs(self):
        ravar_variants = pd.read_csv('data/genome/RAVAR/snp_fulltable.txt', sep='\t')
        kept_columns = ['RSID', 'CHR', 'POS', 'Genotype', 'MAF', 'Mapped Gene', 'RID', 'Reported Trait', 'Trait Label']
        ravar_variants = ravar_variants[kept_columns].drop_duplicates()
        for trait in set(ravar_variants['Trait Label']):
            self.disease2rare_variants[trait] = set(ravar_variants[ravar_variants['Trait Label'] == trait]['RSID'])
            self.disease2rare_variants['All_disease'] |= self.disease2rare_variants[trait]
        Saver(self.disease2rare_variants, self.ravar_rs_path).save_pkl()


def process_gene_info():
    # download from genecode: gencode.v49.annotation.gtf
    gene_df = pd.read_csv('/data/projects/xuy/DeepMEP/data/genome/geneinfo/gencode.v49.annotation.gtf',
                          skiprows=5, sep='\t', header=None)
    gene_df = gene_df[[0, 2, 3, 4, 8]]
    gene_df.columns = ['chr', 'type', 'start', 'end', 'gene_id']
    gene_df = gene_df[gene_df['type'] == 'gene']
    gene_df = gene_df[gene_df['chr'].isin(set([f'chr{i}' for i in range(1, 23)]))]

    def func(x):
        return x.split('gene_name ')[1].split(';')[0].strip('"')

    gene_df['gene_id'] = gene_df['gene_id'].apply(func)
    gene_df = gene_df.drop(columns=['type']).reset_index(drop=True)
    gene_df['gene_length'] = gene_df['end'] - gene_df['start']
    i_lst = []
    for rsid, group in gene_df.groupby('gene_id'):
        i_lst.append(group['gene_length'].idxmax())
    gene_df = gene_df.iloc[i_lst].reset_index(drop=True)
    gene_df.to_csv('/data/projects/xuy/DeepMEP/data/genome/geneinfo/gene_info.txt', sep='\t', index=False)


class GWASMafLdProcessor:
    def __init__(self):
        self.work_dir = 'data/genome/enrichment'
        self.variantid2rsid = CommonSNP().variantid2rsid()
        self.variant_type = 'common'
        self.maf_ld_mat = self.get_maf_ld_mat()
        self.tag2ld_snp = self.get_tag2ld_snp()

    def get_tag2ld_snp(self) -> pd.DataFrame:
        file_path = f'{self.work_dir}/maf/EUR_maf0.005/{self.variant_type}_tag2ld_snps.txt'
        if os.path.exists(file_path):
            return pd.read_csv(file_path, sep='\t')
        else:
            path1 = f'{self.work_dir}/maf/EUR_maf0.005/raw/EUR_maf0.005_r2_0.8.txt'
            path2 = 'data/genome/enrichment/maf/EUR_maf0.005/raw/EUR_maf0.005_pruned_0.8_output.prune.in'
            ld_0_8_snps = pd.read_csv(path1, sep='\t')
            pruned_snps = set(pd.read_csv(path2, header=None).iloc[:, 0])  # 3568192
            pruned_snps1 = pruned_snps & set(ld_0_8_snps['SNP_A'])
            pruned_snps2 = (pruned_snps - set(ld_0_8_snps['SNP_A'])) & set(ld_0_8_snps['SNP_B'])
            pruned_snps3 = list(pruned_snps)
            temp_1 = ld_0_8_snps.loc[ld_0_8_snps['SNP_A'].isin(pruned_snps1), ['SNP_A', 'SNP_B', 'R2']]
            temp_2 = ld_0_8_snps.loc[
                ld_0_8_snps['SNP_B'].isin(pruned_snps2), ['SNP_B', 'SNP_A', 'R2']].rename(
                columns={'SNP_A': 'SNP_B', 'SNP_B': 'SNP_A'}
            )
            temp_3 = pd.DataFrame(
                zip(pruned_snps3, pruned_snps3, [1.0] * len(pruned_snps3)), columns=['SNP_A', 'SNP_B', 'R2']
            )
            rs = pd.concat([temp_1, temp_2, temp_3], axis=0).rename(columns={'SNP_A': 'Tag', 'SNP_B': 'LD'})  # 4335904
            rs = rs.loc[rs['Tag'].isin(self.variantid2rsid.keys()), :]  # 1938850
            rs = rs.loc[rs['LD'].isin(self.variantid2rsid.keys()), :]  # 1643781
            tag = [self.variantid2rsid[v] for v in rs['Tag']]
            ld = [self.variantid2rsid[v] for v in rs['LD']]
            rs = pd.DataFrame({'Tag': tag, 'LD': ld, 'R2': rs['R2'].to_list()})  # 1643781
            rs.to_csv(file_path, sep='\t', index=False)
            return rs

    def get_maf_ld_mat(self) -> pd.DataFrame:
        # generating the ld_score and maf
        # return: [SNP, chr, bp, MAF, mean_rsq, snp_num, max_rsq, ldscore]
        # pwd: /data/g_gamazon_lab/zhoud2/PTMDeep
        # bfile: /data/coxvgi/zhoud2/data/kgp/phase3/from_plink_38/sub_pop_processed/EUR_maf0.005
        # 1. this cmd computes the ld-score and maf
        # gcta64 --bfile EUR_maf0.005 --ld-score --ld-wind 1000 --ld-rsq-cutoff 0.01 --out ldscore
        # 2. this cmd removes one of the snp pair whose r2 > 0.8
        # plink --bfile EUR_maf0.005 --indep-pairwise 50 5 0.8 --out EUR_maf0.005_pruned_0.8_output
        # 3. this cmd computes the high correlated snp pairs (r2 > 0.8)
        # plink --bfile EUR_maf0.005 --r2 --ld-window-r2 0.8 --ld-window 99999 --ld-window-kb 1000 --out EUR_maf0.005_r2_0.8
        file_path = f'{self.work_dir}/maf/EUR_maf0.005/{self.variant_type}_maf_ld_all.txt'
        if os.path.exists(file_path):
            return pd.read_csv(file_path, sep='\t')
        else:
            maf_ld = pd.read_csv(f'{self.work_dir}/maf/ldscore.score.txt', sep='\t')
            maf_ld = maf_ld[maf_ld['SNP'].isin(self.variantid2rsid.keys())]  # 584425
            rsid_lst = [self.variantid2rsid[v] for v in maf_ld['SNP']]
            maf_ld['rsid'] = rsid_lst
            maf_ld = maf_ld.rename(columns={'SNP': 'rsid', 'rsid': 'SNP'})
            maf_ld.to_csv(file_path, sep='\t', index=False)
            return maf_ld

    def split_maf_ld_mat(self, full_set: Set, full_set_name: str) -> Dict:
        # split maf_ld_mat into 25 parts
        file_path = f'{self.work_dir}/maf/EUR_maf0.005/{self.variant_type}_{full_set_name}_split_maf_ld.pkl'
        if os.path.exists(file_path):
            return Loader(file_path).load_pkl()
        else:
            split_dic = {}
            mat = self.maf_ld_mat[self.maf_ld_mat['SNP'].isin(full_set)]
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
            maf_cutoff = [mat['MAF'].quantile(i) for i in quantiles]
            ld_cutoff = [mat['ldscore'].quantile(i) for i in quantiles]
            for i in range(5):
                maf_bool = (mat['MAF'] >= maf_cutoff[i]) & (mat['MAF'] < maf_cutoff[i + 1])
                temp_df = mat.loc[maf_bool, :]
                for j in range(5):
                    ld_bool = (mat['ldscore'] >= ld_cutoff[j]) & (mat['ldscore'] < ld_cutoff[j + 1])
                    part_ids = set(temp_df.loc[maf_bool & ld_bool, 'SNP'])
                    split_dic[f'maf_{i}_ld_{j}'] = part_ids
            Saver(split_dic, file_path).save_pkl()
            return split_dic
