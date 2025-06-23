from utils import *
from Protein import ProteinSeq


class ReferenceGenome:
    def __init__(self):
        # generate the dictionary which can obtain reference DNA sequence from chromosome
        # return: {'chr1': 'AAAGGCTTT...', ...}
        cache_path = 'data/genome/ref_genome/ref_genome.pkl'
        if os.path.exists(cache_path):
            ref_genome: Dict = Loader(cache_path).load_pkl()
        else:
            fasta_file = 'data/genome/ref_genome/GRCh38.primary_assembly.genome.fa'
            chrs = [
                'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
                'chr13',
                'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM'
            ]
            parse_file = Bio.SeqIO.to_dict(Bio.SeqIO.parse(fasta_file, "fasta"))
            ref_genome = {chro: str(parse_file[chro].seq) for chro in chrs}
            Saver(ref_genome, cache_path).save_pkl()
        self.ref_genome = ref_genome


class CommonSNP:
    def __init__(self):
        # process the snp annotation file
        # return: DataFrame, ['chr', 'bp', 'rsid', 'ref', 'alt']
        cache_path = 'data/genome/common_snp/snp_annotation.txt'
        if os.path.exists(cache_path):
            snp_annotation = pd.read_csv(cache_path, sep='\t')
        else:
            snp_file = 'data/genome/common_snp/00-common_all.vcf'
            raw_snp_annotation = pd.read_csv(filepath_or_buffer=snp_file, sep='\t', comment='#', header=None, dtype=str)
            snp_annotation = raw_snp_annotation.iloc[:, 0:5].copy()
            snp_annotation.columns = ['chr', 'bp', 'rsid', 'ref', 'alt']
            snp_annotation.loc[:, 'chr'] = 'chr' + snp_annotation['chr']
            ref_alt = zip(snp_annotation['ref'], snp_annotation['alt'])
            single_variant = [self.is_single_variant(ref, alt) for ref, alt in ref_alt]
            snp_annotation = snp_annotation.loc[single_variant, :]
            snp_annotation.to_csv(cache_path, index=False, sep='\t')
        self.snp_annotation = snp_annotation

    def rsid2variantid(self) -> Dict:
        # return: {'rsid': '1:17556:C:T', ...}
        file_path = 'data/genome/common_snp/rsid2variantid.pkl'
        if os.path.exists(file_path):
            return Loader(file_path).load_pkl()
        else:
            snp_df = self.snp_annotation
            ids = {}
            for i in tqdm(range(snp_df.shape[0])):
                rsid = snp_df.iloc[i, 2]
                chr_num = snp_df.iloc[i, 0].split("chr")[-1]
                bp = snp_df.iloc[i, 1]
                ref = snp_df.iloc[i, 3]
                if ',' in ref:
                    ref = ''.join(ref.split(','))
                alt = snp_df.iloc[i, 4]
                if ',' in alt:
                    alt = ''.join(alt.split(','))
                ids[rsid] = f'{chr_num}:{bp}:{ref}:{alt}'
            Saver(ids, file_path).save_pkl()
            return ids

    def variantid2rsid(self) -> Dict:
        # return: {'1:17556:C:T': 'rsid', ...}
        variantid2rsid = {}
        rsid2variantid = self.rsid2variantid()
        for rsid, variantid in rsid2variantid.items():
            variantid2rsid[variantid] = rsid
        return variantid2rsid

    def get_snp_chr_pos_allele(self, rsid) -> Tuple[str, int, str, str]:
        row = self.snp_annotation.loc[self.snp_annotation['rsid'] == rsid, :]
        # (chr, pos, ref, alt)
        return row.iloc[0, 0], int(row.iloc[0, 1]), row.iloc[0, 3], row.iloc[0, 4]

    @staticmethod
    def is_single_variant(ref: str, alt: str) -> bool:
        # judge a snp whether is a single variant
        if len(ref) == 1:
            if len(alt) == 1:
                return True
            elif ',' in alt:
                alt_lst = alt.split(',')
                if len(alt_lst) + alt.count(',') == len(alt):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
