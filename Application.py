from utils import *
from Model import *
from DataProcess import *
from Protein import Embedding


def get_final_model(data_from, window_size, embedding_layer):
    model_path = (f'rs/our_model/{data_from}/prediction_esm-embedding/'
                  f'{window_size}_{embedding_layer}/model_{window_size}_{embedding_layer}.pth')
    model = Loader(model_path).load_pth().cpu()
    model_type: Literal['classification', 'regression']
    labels_num, drop_p, model_type = 2, [0.4, 0.15, 0.1], 'classification'
    if data_from == 'uniprot':
        labels_num, drop_p, model_type = 2, [0.4, 0.15, 0.1], 'classification'
    elif data_from == 'cbmap':
        labels_num, drop_p, model_type = 1, [0.5, 0.5, 0.15], 'regression'
    elif data_from == 'epsd_family':
        labels_num, drop_p, model_type = 9, [0.5, 0.5, 0.15], 'classification'
    new_model = BasicPredictor(seq_len=2 * window_size + 1, conv_in_channels=320, mlp_mediate_neuron_num=512,
                               mlp_feature_num=12, labels_num=labels_num, drop_p=drop_p, model_type=model_type)
    new_model.integrate_layers = model.integrate_layers
    new_model.out_layer = model.out_layer
    return new_model


class Utilizer:
    def __init__(self, data_from: Literal['uniprot', 'cbmap', 'epsd_family', 'epsd_group'],
                 device_id: Optional[int] = None):
        self.device_id = device_id
        self.data_from = data_from
        self.embedder = Embedding(device_id=device_id, pool_size=1, model_version=320)

    def single_model_predict_phos(self, seq_lst: List, window_size: int, embedding_layer: int) -> torch.Tensor:
        model = get_final_model(self.data_from, window_size, embedding_layer).to(get_device(self.device_id))
        pred_p, batch_size = [], 5120
        for i in range(math.ceil(len(seq_lst) / batch_size)):
            batch_seq = seq_lst[i * batch_size: (i + 1) * batch_size]
            batch_embedding = self.embedder.bulk_embedding(seq_lst=batch_seq, return_layer=embedding_layer)
            pred_p.append(model.predict(batch_embedding.transpose(1, 2)))
        pred_p = torch.cat(pred_p)
        return pred_p

    def ensemble_model_predict_phos(self, seq_lst: List, window_size: int) -> torch.Tensor:
        pred_p_lst = []
        for embedding_layer in range(1, 7):
            pred_p_lst.append(self.single_model_predict_phos(seq_lst, window_size, embedding_layer))
        pred_p = torch.mean(torch.stack(pred_p_lst, dim=-1), dim=-1)
        return pred_p


def generate_phosphoration_prediction(data_from: Literal['uniprot', 'cbmap'], device_id: int):
    window_size = 15 if data_from == 'uniprot' else 12
    datasetor = TrainingDataset(window_size=window_size, use_structure=False, data_from=data_from)
    seq_instance = datasetor.getSeqInstance()
    label_instances, context_instances, site_id = seq_instance[0], seq_instance[1], seq_instance[-1]
    if data_from == 'uniprot':
        label_instances = label_instances[:, window_size, 1]
    else:
        label_instances = label_instances[:, window_size]
    pos_flanking = [s[window_size-7: window_size+8] for s in context_instances]
    pred_p = Utilizer(data_from, device_id).ensemble_model_predict_phos(context_instances, window_size).cpu().detach().numpy()[:, -1]
    pd.DataFrame({'Peptide': pos_flanking, 'Pred.P': pred_p, 'Site': site_id, 'Label': label_instances}).to_csv(
        f'rs/snp_prioritize/PTM_SEA/{data_from}/pred_p.txt', sep='\t', index=False)


class Applier:
    def __init__(self, data_from: Literal['uniprot', 'cbmap', 'epsd_family', 'epsd_group'],
                 device_id: Optional[int] = None):
        self.data_from = data_from
        self.device_id = device_id
        self.utilizor = Utilizer(data_from, device_id)
        if data_from == 'uniprot':
            window_size = 15
        elif data_from == 'cbmap':
            window_size = 12
        else:
            window_size = 7
        self.window_size = window_size
        self.embedding_layer = 'ensemble'
        self.apply_dataset = ApplicationDataset()

    def apply_model(self, batch_size: int = 40960):
        rs_save_dir = f'rs/snp_prioritize/p_diff/{self.data_from}_{self.window_size}_{self.embedding_layer}'
        make_dir(rs_save_dir)

        # step 1: get application data
        print('Step 1: Getting the application dataset!')
        application_instance = self.apply_dataset.getApplicationInstances(window_size=self.window_size)
        normal_app_i = application_instance[application_instance['sty_changed'] == 'unchange']
        sty2aa_app_i = application_instance[application_instance['sty_changed'] == 'sty2aa']
        aa2sty_app_i = application_instance[application_instance['sty_changed'] == 'aa2sty']
        normal_ref_instances, normal_alt_instances = normal_app_i['ref_seq'].to_list(), normal_app_i['alt_seq'].to_list()
        sty2aa_ref_instances = sty2aa_app_i['ref_seq'].to_list()
        aa2sty_alt_instances = aa2sty_app_i['alt_seq'].to_list()

        # step 2: apply model
        print('Step 2: Computing the P_ref, P_alt of the influenced sites under the scale of missense mutation!')
        rs_instance_path = f'{rs_save_dir}/p_diff_by_instance.csv'
        # embedder = Embedding(device_id=self.device_id, pool_size=1, model_version=320)
        if not os.path.exists(rs_instance_path):
            # step 2.1: process normal mutation
            part_ref_p_lst = [self.utilizor.ensemble_model_predict_phos(normal_ref_instances, self.window_size)]
            part_alt_p_lst = [self.utilizor.ensemble_model_predict_phos(normal_alt_instances, self.window_size)]

            # step 2.2: process sty -> aa
            sty2aa_ref_p = self.utilizor.ensemble_model_predict_phos(sty2aa_ref_instances, self.window_size)
            sty2aa_alt_p = torch.zeros(sty2aa_ref_p.shape, device=get_device(self.device_id)) + torch.min(sty2aa_ref_p)
            part_ref_p_lst.append(sty2aa_ref_p)
            part_alt_p_lst.append(sty2aa_alt_p)

            # step 2.3: process aa -> sty
            aa2sty_alt_p = self.utilizor.ensemble_model_predict_phos(aa2sty_alt_instances, self.window_size)
            aa2sty_ref_p = torch.zeros(aa2sty_alt_p.shape, device=get_device(self.device_id)) + torch.min(aa2sty_alt_p)
            part_ref_p_lst.append(aa2sty_ref_p)
            part_alt_p_lst.append(aa2sty_alt_p)
            ref_p, alt_p = torch.cat(part_ref_p_lst), torch.cat(part_alt_p_lst)
            if self.data_from == 'epsd_family':
                ref_p, alt_p = ref_p[:, :, 1], alt_p[:, :, 1]
                ref_p, alt_p = torch.sum(ref_p, dim=-1), torch.sum(alt_p, dim=-1)
            else:
                ref_p, alt_p = ref_p[:, -1], alt_p[:, -1]
            p_diff_lst = torch.abs(ref_p - alt_p).tolist()
            ref_p_lst = ref_p.tolist()
            alt_p_lst = alt_p.tolist()
            i_lst = list(range(application_instance.shape[0]))

            def _bind_(_col_):
                return normal_app_i[_col_].to_list() + sty2aa_app_i[_col_].to_list() + aa2sty_app_i[_col_].to_list()

            snp_lst, d_lst, gene_lst = _bind_('SNP'), _bind_('variant_sty_distance'), _bind_('gene')
            sty_changed_lst = _bind_('sty_changed')

            rs_instance = pd.DataFrame(zip(
                i_lst, snp_lst, d_lst, gene_lst, sty_changed_lst, ref_p_lst, alt_p_lst, p_diff_lst
            ), columns=['i', 'SNP', 'variant_sty_distance', 'gene_name', 'STY_change', 'p_ref', 'p_alt', 'p_diff'])
            rs_instance = rs_instance.merge(self.apply_dataset.seq_data[['SNP', 'aa_ref', 'aa_alt', 'ref', 'alt']], on='SNP')
            rs_instance = rs_instance.rename(columns={'ref': 'base_ref', 'alt': 'base_alt'})
            new_columns = [
                'i', 'SNP', 'aa_ref', 'aa_alt', 'base_ref', 'base_alt', 'gene_name',
                'variant_sty_distance', 'STY_change', 'p_ref', 'p_alt', 'p_diff'
            ]
            rs_instance = rs_instance[new_columns]
            self.plot_difference_distance(rs_instance, filename=f'{rs_save_dir}/p_diff_distance.png')
            self.plot_P_dist(rs_instance, filename=f'{rs_save_dir}/p_distribution.pdf')
            rs_instance.to_csv(path_or_buf=rs_instance_path, sep=',', index=False)
        else:
            rs_instance = pd.read_csv(rs_instance_path)

        # step 3: select the max difference
        print('Step 3: Selecting the max difference between P_ref and P_alt for each missense SNP!')
        max_p_diff_path = f'{rs_save_dir}/max_p_diff.csv'
        if not os.path.exists(max_p_diff_path):
            i_lst = []
            for rsid, group in rs_instance.groupby('SNP'):
                max_effect_row = group.loc[group['p_diff'].idxmax()]
                i_lst.append(max_effect_row['i'])
            rs_snp_by_p = rs_instance[rs_instance['i'].isin(i_lst)]
            rs_snp_by_p.to_csv(path_or_buf=max_p_diff_path, index=False)
        else:
            rs_snp_by_p = pd.read_csv(max_p_diff_path)
        self.plot_difference_distribution(
            p_diff=rs_snp_by_p['p_diff'], filename=f'{rs_save_dir}/max_p_diff_distribution.pdf'
        )
        print('The application process has been completed!')

    @staticmethod
    def plot_P_dist(rs_instance: pd.DataFrame, filename: str):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(rs_instance['p_ref'], bins=20, kde=True, color='blue', edgecolor='black')
        plt.xlabel('P_ref')
        plt.ylabel('Frequency')
        plt.title('Distribution of P_ref')
        plt.subplot(1, 2, 2)
        sns.histplot(rs_instance['p_alt'], bins=20, kde=True, color='blue', edgecolor='black')
        plt.xlabel('P_alt')
        plt.ylabel('Frequency')
        plt.title('Distribution of P_alt')
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_difference_distance(rs_instance: pd.DataFrame, filename: str):
        d = rs_instance['variant_sty_distance'].abs()
        plt.figure(figsize=(12, 12))
        plt.scatter(d, rs_instance['p_diff'], marker='o', s=1)
        plt.title('P difference and SNP distance', fontsize=20)
        plt.ylabel('P difference', fontsize=18)
        plt.xlabel('SNP distance', fontsize=18)
        # plt.legend()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_difference_distribution(p_diff, filename):
        plt.figure(figsize=(6, 6), facecolor='white')
        sns.histplot(p_diff, bins=20, kde=True, color='blue', edgecolor='black')
        plt.xlabel('|P_ref - P_alt|')
        plt.ylabel('Frequency')
        plt.title('Distribution of differences between P')
        plt.savefig(filename)
        plt.close()


class EnrichAnalyzer:
    def __init__(self, data_from: Literal['uniprot', 'cbmap', 'epsd_family', 'epsd_group'],
                 variant_type: Literal['common', 'rare'],
                 sample_type: Literal['random', 'distribution'],
                 full_set: Literal['ALL', 'Missense'] = 'Missense', is_print: bool = True,
                 ptm_sig_threshold_q: float = 0.5, repeat_times: int = 10000):
        assert (variant_type == 'rare' and sample_type == 'random') or variant_type == 'common'
        self.data_from = data_from
        self.repeat_times = repeat_times
        self.sample_type = sample_type
        self.variant_type = variant_type
        self.full_set = 'ALL' if ptm_sig_threshold_q == 0. else full_set
        self.data_dir = 'data/genome/enrichment'
        now_time = datetime.datetime.now()
        rs_dir = f'rs/snp_prioritize/enrichment/{full_set}_{sample_type}_{variant_type}_{data_from}_{repeat_times}'
        make_dir(rs_dir)
        self.rs_dir = f'{rs_dir}/{ptm_sig_threshold_q}'
        make_dir(self.rs_dir)
        if data_from == 'uniprot':
            self.snp_p_diff_df = pd.read_csv(f'rs/snp_prioritize/p_diff/uniprot_15_ensemble/max_p_diff.csv')
        elif data_from == 'cbmap':
            self.snp_p_diff_df = pd.read_csv(f'rs/snp_prioritize/p_diff/cbmap_12_ensemble/max_p_diff.csv')
        elif data_from == 'epsd_family':
            self.snp_p_diff_df = pd.read_csv(f'rs/snp_prioritize/p_diff/cbmap_7_ensemble/max_p_diff.csv')
        # use quantile as ptm significant threshold, default 0.5
        self.ptm_sig_threshold = self.snp_p_diff_df['p_diff'].quantile(q=abs(ptm_sig_threshold_q))
        self.ptm_mis_snps = set(self.snp_p_diff_df['SNP'])
        if 0 < ptm_sig_threshold_q < 1:
            sig_snps = set(self.snp_p_diff_df[self.snp_p_diff_df['p_diff'] >= self.ptm_sig_threshold]['SNP'])
        elif ptm_sig_threshold_q == 0:
            sig_snps = self.ptm_mis_snps
        elif ptm_sig_threshold_q == 1:
            sty2aa_snps, aa2sty_snps = ApplicationDataset().getSTYSnps()
            sig_snps = sty2aa_snps | aa2sty_snps
        else:
            sig_snps = set(self.snp_p_diff_df[self.snp_p_diff_df['p_diff'] < self.ptm_sig_threshold]['SNP'])

        if variant_type == 'common':
            # GWAS sig snps
            gwas_processor = GWASProcessor()
            self.disease_associated_snps = gwas_processor.gwas_rs
            self.diseases = gwas_processor.get_diseases()
            # MAF LD
            self.gwas_maf_ld_processor = GWASMafLdProcessor()
            maf_snps = set(self.gwas_maf_ld_processor.maf_ld_mat['SNP'])
            other_snps = self.ptm_mis_snps - maf_snps
            self.standard_full_set = maf_snps | self.ptm_mis_snps
            # PTM sig snps
            tag_ld_mat = self.gwas_maf_ld_processor.tag2ld_snp.copy()
            sig_snps_without_tag = sig_snps - set(tag_ld_mat['LD'])
            tag_sig_snps = set(tag_ld_mat.loc[tag_ld_mat['LD'].isin(sig_snps), 'Tag']) | sig_snps_without_tag
            self.ptm_sig_snps = tag_sig_snps & self.standard_full_set
            self.full_split_dic = self.gwas_maf_ld_processor.split_maf_ld_mat(self.standard_full_set, 'ALL')
            self.full_split_dic['Other'] = other_snps
            self.miss_split_dic = self.gwas_maf_ld_processor.split_maf_ld_mat(self.ptm_mis_snps, 'Missense')
            self.miss_split_dic['Other'] = other_snps
        else:
            # ClinVar snps
            clinvar_processor = ClinVarProcessor()
            self.disease_associated_snps = clinvar_processor.disease2rare_variants
            self.diseases = clinvar_processor.get_diseases()
            self.standard_full_set = clinvar_processor.get_all_rare_variants()
            self.ptm_sig_snps = sig_snps & self.standard_full_set
            self.ptm_mis_snps = self.ptm_mis_snps & self.standard_full_set

        # print running parameters
        if is_print:
            print('=======================================RUNNING PARAMETERS=========================================')
            print(f'Model source: {data_from}\nThreshold: {self.ptm_sig_threshold: .3f}\nPermutation Times: {repeat_times}')
            print(f'Full set: {self.full_set}\nSample type: {sample_type}')
            print(f'Standard Full Set Num: {len(self.standard_full_set)}') if variant_type == 'common' else None
            print(f'No. PTM Sig SNPs: {len(self.ptm_sig_snps)}\nNo. all missense SNPs: {len(self.ptm_mis_snps)}\n{self.rs_dir}\n\n')

    def PTMEnrichment(self):
        show_rs = []
        summary_rs = {
            'Disease': self.diseases, 'Disease_associated_missense': [], self.full_set: [],
            'Intersect_Num': [0] * len(self.diseases), 'Mean_Rand_Intersect_Num': [], 'Fold_change': [],
            'Threshold': [self.ptm_sig_threshold] * len(self.diseases)
        }
        plot_data = []
        for i, disease_name in enumerate(self.diseases):
            disease_rs = self.single_disease_ptm_enrichment(i)
            summary_rs['Disease_associated_missense'].append(
                len(self.disease_associated_snps[disease_name] & self.ptm_mis_snps)
            )
            summary_rs[self.full_set].append(disease_rs[self.full_set]['plot_data'][0])
            i_num = disease_rs[self.full_set]['plot_data'][1]
            mean_rand_i_num = np.mean(disease_rs[self.full_set]['plot_data'][2])
            if mean_rand_i_num == 0.:
                mean_rand_i_num += 0.01
            fold_change = round(i_num / mean_rand_i_num, 2)
            summary_rs['Intersect_Num'][i] = i_num
            summary_rs['Mean_Rand_Intersect_Num'].append(round(mean_rand_i_num, 2))
            summary_rs['Fold_change'].append(fold_change)
            show_rs.append(pd.DataFrame(disease_rs[self.full_set]['show_rs']))
            plot_data.append(disease_rs[self.full_set]['plot_data'])
        self.plotPTMEnrichment(filename=f'{self.rs_dir}/{self.full_set}_figure.pdf', plot_data=plot_data, show=False)

        pd.DataFrame(summary_rs).rename(columns={self.full_set: 'P'}).to_csv(
            f'{self.rs_dir}/summary_rs.csv', index=False)
        show_rs = pd.concat(show_rs, axis=0)
        show_rs.to_csv(f'{self.rs_dir}/output.txt', sep='\t', index=False)

    def single_disease_ptm_enrichment(self, disease_ii: int) -> Dict:
        disease = self.diseases[disease_ii]
        save_path = f'{self.rs_dir}/parallel/{disease}.pkl'
        make_dir(f'{self.rs_dir}/parallel/')
        if os.path.exists(save_path):
            return Loader(save_path).load_pkl()
        else:
            disease_rs = {}
            disease_snp_set = self.disease_associated_snps[disease] & self.standard_full_set
            intersect_snp, intersect_num, p_value, rand_intersect_lst = self.permutation_test(
                self.ptm_sig_snps, disease_snp_set
            )
            disease_rs[self.full_set] = {
                'show_rs': self.enrichment_output(intersect_snp_set=intersect_snp, disease=f'{disease}_{self.full_set}'),
                'plot_data': (p_value, intersect_num, rand_intersect_lst, disease, self.full_set, self.ptm_sig_threshold)
            }
            Saver(disease_rs, save_path).save_pkl()
            return disease_rs

    def permutation_test(self, sig_mis_set, dis_asso_set):
        # return: (intersect_snps, intersect_num, p_value, permutation_lst)
        intersect_num = len(sig_mis_set & dis_asso_set)
        i, permutation_lst = 0, []
        for _ in tqdm(range(self.repeat_times), desc='Permutation Test'):
            rand_set = self.sample_snps(sig_mis_set)
            rand_intersect_num = len(rand_set & dis_asso_set)
            permutation_lst.append(rand_intersect_num)
            if rand_intersect_num >= intersect_num:
                i += 1
        return sig_mis_set & dis_asso_set, intersect_num, i / self.repeat_times, permutation_lst

    def sample_snps(self, snp_set: Set) -> Set:
        if self.sample_type == 'distribution':
            rand_set = set()
            split_dic = self.full_split_dic if self.full_set == 'ALL' else self.miss_split_dic
            for i in range(5):
                for j in range(5):
                    key_str = f'maf_{i}_ld_{j}'
                    rand_set.update(set(random.sample(
                        split_dic[key_str], len(split_dic[key_str] & snp_set)
                    )))
            rand_set.update(set(random.sample(split_dic['Other'], len(split_dic['Other'] & snp_set))))
        else:
            full_set = self.standard_full_set if self.full_set == 'ALL' else self.ptm_mis_snps
            rand_set = set(random.sample(list(full_set), len(snp_set)))
        return rand_set

    def enrichment_output(self, intersect_snp_set: Set, disease: str) -> pd.DataFrame:
        # rsid, disease, |Palt-Pref|, protein, alt_AA, ref_AA
        if self.variant_type == 'common':
            tag_ld_mat = self.gwas_maf_ld_processor.tag2ld_snp.copy()
            intersect_snp_set_without_tag = intersect_snp_set - set(tag_ld_mat['Tag'])
            ld_snps = set(tag_ld_mat.loc[tag_ld_mat['Tag'].isin(intersect_snp_set), 'LD']) | intersect_snp_set_without_tag
            select_df = self.snp_p_diff_df[self.snp_p_diff_df['SNP'].isin(ld_snps)].copy()
            select_df['disease'] = disease
            out_put = select_df.merge(tag_ld_mat[['Tag', 'LD']], left_on='SNP', right_on='LD', how='left').to_dict('list')
        else:
            select_df = self.snp_p_diff_df[self.snp_p_diff_df['SNP'].isin(intersect_snp_set)].copy()
            select_df['disease'] = disease
            out_put = select_df.to_dict('list')
        return out_put

    def plotPTMEnrichment(self, filename, plot_data: List[Tuple[float, int, List, str, str, float]],
                          diseases=None, ncol=4, show=True):
        # called by self.ptm_enrichment
        if diseases is None:
            diseases = self.diseases
        nrow, p_i = math.ceil(len(diseases) / ncol), 1
        plt.figure(figsize=(ncol * 6, nrow * 6), facecolor='white')
        for p_value, intersect_num, rand_intersect_lst, disease, full_set, threshold in plot_data:
            if disease in diseases:
                plt.subplot(nrow, ncol, p_i)
                sns.histplot(
                    rand_intersect_lst, bins=20, kde=True, color='green', edgecolor='black',
                    label=f'Random Intersect Num'
                )
                plt.axvline(x=intersect_num, linestyle='--', color='red', label=f'Intersect Num: {intersect_num}')
                plt.title(f'{disease}--P: {p_value}')
                plt.xlabel('Permutation Statistics')
                plt.ylabel('Density')
                plt.legend(loc='upper right')
                p_i += 1
        plt.savefig(filename)
        plt.show() if show else None
        plt.close()
