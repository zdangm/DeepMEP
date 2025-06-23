from utils import *
import freesasa


class ResidueInfo:
    def __init__(self):
        self.__AAs__: List[str] = [
            'A', 'P', 'E', 'G', 'S', 'D', 'R', 'Q', 'K', 'L', 'T', 'C',
            'M', 'V', 'H', 'Y', 'F', 'I', 'N', 'W', 'X', 'U', '_'
        ]
        self.__atomMass__: Dict[str, int] = {'C': 12, 'N': 14, 'O': 16, 'S': 32, 'P': 31}
        self.__three2one__: Dict[str, str] = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'SEC': 'U', 'ANY': 'X'
        }
        self.__atomNum__: Dict[str, int] = {
            'A': 5, 'C': 6, 'D': 8, 'E': 9, 'F': 11, 'G': 4, 'H': 10, 'I': 8,
            'K': 9, 'L': 8, 'M': 8, 'N': 8, 'P': 7, 'Q': 9, 'R': 11, 'S': 6,
            'T': 7, 'V': 7, 'W': 14, 'Y': 12, 'X': 0, 'U': 8
        }
        self.__chainChargeNum__: Dict[str, int] = {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1, 'I': 0,
            'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1, 'S': 0,
            'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'X': 0, 'U': 0
        }
        self.__sideChainHydrogenBondNum__: Dict[str, int] = {
            'A': 2, 'C': 2, 'D': 4, 'E': 4, 'F': 2, 'G': 2, 'H': 4, 'I': 2,
            'K': 2, 'L': 2, 'M': 2, 'N': 4, 'P': 2, 'Q': 4, 'R': 4, 'S': 4,
            'T': 4, 'V': 2, 'W': 3, 'Y': 3, 'X': 0, 'U': 1
        }
        self.__sideChainPKa__: Dict[str, float] = {
            'A': 7.0, 'C': 7.0, 'D': 3.65, 'E': 3.22, 'F': 7.0, 'G': 7.0, 'H': 6.0, 'I': 7.0, 'K': 10.53,
            'L': 7.0, 'M': 7.0, 'N': 8.18, 'P': 7.0, 'Q': 7.0, 'R': 12.48, 'S': 7.0, 'T': 7.0, 'V': 7.0,
            'W': 7.0, 'Y': 10.07, 'X': 0, 'U': 5.35
        }
        self.__Hydrophobicity__: Dict[str, float] = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': 3.9,
            'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
            'W': -0.9, 'Y': -1.3, 'X': 0, 'U': -0.6
        }
        self.__secondStructureDic__: Dict[str, List[int]] = {
            'H': [1, 0, 0, 0, 0, 0, 0, 0],  # alpha-helix
            'B': [0, 1, 0, 0, 0, 0, 0, 0],  # beta-sheet
            'E': [0, 0, 1, 0, 0, 0, 0, 0],  # beta-turn
            'G': [0, 0, 0, 1, 0, 0, 0, 0],  # gamma-turn
            'I': [0, 0, 0, 0, 1, 0, 0, 0],  # Random coil
            'T': [0, 0, 0, 0, 0, 1, 0, 0],  # turn
            'S': [0, 0, 0, 0, 0, 0, 1, 0],  # loop
            '-': [0, 0, 0, 0, 0, 0, 0, 1]  # no structure
        }

    def getSecondStructureDic(self) -> Dict[str, List[int]]:
        return self.__secondStructureDic__

    def getAtomMass(self) -> Dict[str, int]:
        return self.__atomMass__

    def getAAs(self) -> List[str]:
        return self.__AAs__

    def getOneHotDic(self) -> Dict[str, List]:
        aa2vec = {}
        i = 0
        for aa in self.__AAs__:
            if aa in {'X', 'U'}:
                aa2vec[aa] = [0.05] * (len(self.__AAs__) - 3)
            else:
                baseVec = [0] * (len(self.__AAs__) - 3)
                if aa != '_':
                    baseVec[i] = 1
                    i += 1
                aa2vec[aa] = baseVec
        return aa2vec

    def getBlosumDic(self) -> Dict[str, List]:
        import blosum
        aa2vec = {}
        blosum62 = blosum.BLOSUM(62)
        for aa in self.__AAs__:
            baseVec = [0] * (len(self.__AAs__) - 1)
            if aa != '_':
                i = 0
                for m_aa in self.__AAs__:
                    if m_aa != '_':
                        baseVec[i] = blosum62[aa][m_aa]
                        i += 1
            aa2vec[aa] = baseVec
        return aa2vec

    def getThree2One(self) -> Dict[str, str]:
        return self.__three2one__

    def getAtomNum(self) -> Dict[str, int]:
        return self.__atomNum__

    def getChainChargeNum(self) -> Dict[str, int]:
        return self.__chainChargeNum__

    def getSideChainHydrogenBondNum(self) -> Dict[str, int]:
        return self.__sideChainHydrogenBondNum__

    def getSideChainPKa(self) -> Dict[str, float]:
        return self.__sideChainPKa__

    def getHydrophobicity(self) -> Dict[str, float]:
        return self.__Hydrophobicity__


class ProteinSeq:
    def __init__(self):
        cache_path = 'data/protein_sequences.pkl'
        if os.path.exists(cache_path):
            protein_sequences = Loader(cache_path).load_pkl()
        else:
            fasta_file = 'data/dataset/uniprot/02human_uniprotkb_proteome_UP000005640_2023_09_28.fasta'
            protein_sequences = {}
            for record in SeqIO.parse(fasta_file, 'fasta'):
                accession_id = record.id.split('|')[1]
                protein_sequences[accession_id] = str(record.seq)
        self.protein_sequences = protein_sequences
        self.cache_path = cache_path

    def revise_sequences(self, entry_name, new_sequence):
        self.protein_sequences[entry_name] = new_sequence
        Saver(self.protein_sequences, self.cache_path).save_pkl()


class Embedding:
    def __init__(self,
                 device_id: Union[int, Literal['cpu']], # device used for embedding
                 pool_size: int, # the stride length of pooling
                 model_version: Literal[320, 640, 1280, 2560, 5120] # the embedding dimension of different versions of models
                 ):
        self.device = get_device(device_id)
        import esm
        # /home/zhoudan/.cache/torch/hub/checkpoints
        if model_version == 320:
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 320
        elif model_version == 640:
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D() # 640
        elif model_version == 1280:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 1280
        elif model_version == 2560:
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # 2560
        elif model_version == 5120:
            model, alphabet = esm.pretrained.esm2_t48_15B_UR50D() # 5120
        else:
            model, alphabet = None, None
        self.embedding_model = model.eval().to(self.device)
        self.batch_converter = alphabet.get_batch_converter()
        self.pool = pool_size > 1
        self.AvgPool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def mini_batch_embedding(self, split_seq: Any, return_layer: int) -> torch.Tensor:
        # seq_: [(label, sequence), ..., (label, sequence)]
        # seq_: [sequence, sequence, ...]
        # where sequences padded by '<pad>'
        if isinstance(split_seq[0], str):
            split_seq = [(i, split_seq[i]) for i in range(len(split_seq))]
        _, _, split_batch_tokens = self.batch_converter(split_seq)  # batch, seq, ..
        with torch.no_grad():
            results = self.embedding_model(split_batch_tokens.to(self.device), repr_layers=[return_layer], return_contacts=False)
        split_token_representations = results["representations"][return_layer]  # batch, seq, 320
        split_token_representations = split_token_representations[:, 1:-1, :]
        if self.pool:
            split_token_representations = self.AvgPool(split_token_representations)  # batch, seq, 320/pool_size
        return split_token_representations

    def bulk_embedding(self, seq_lst: Any, return_layer: int = 6) -> torch.Tensor:
        batch_size = 100
        batches = math.ceil(len(seq_lst) / batch_size)
        if isinstance(seq_lst[0], tuple):
            seq_lst = [i.replace('_', '<pad>') for _, i in seq_lst]
        else:
            seq_lst = [i.replace('_', '<pad>') for i in seq_lst]
        embedding = torch.cat([
            self.mini_batch_embedding(seq_lst[i * batch_size: (i + 1) * batch_size], return_layer)
            for i in range(batches)
        ], dim=0)
        return embedding


class DiscreteEncoder:
    def __init__(self, add_pos):
        self.Residue = ResidueInfo()
        self.add_pos = add_pos

    def __encoding__(self, seq, aa2vec, process_name):
        # seq: [(label, sequence), ..., (label, sequence)]
        matrice = []
        seq_label = []
        for sequence in tqdm(seq, desc=process_name):
            try:
                seq_label.append(sequence[0])
                if sequence[1] == '':
                    continue
                seq_mat = np.array([[aa2vec[aa] for aa in sequence[1]]])
                matrice.append(seq_mat)
            except Exception as e:
                print(e)
                print(sequence)
                break
        # print(matrice[281])
        mat = np.concatenate(matrice, axis=0)
        print(f'array shape: {mat.shape}')
        pos_len, channel_len = mat.shape[1], mat.shape[2]
        if self.add_pos:
            return np.stack(seq_label, axis=0), mat + addPos(pos_len=pos_len, channel_len=channel_len)
        else:
            return np.stack(seq_label, axis=0), mat

    def one_hot(self, seq):
        return self.__encoding__(seq, self.Residue.getOneHotDic(), 'One-Hot Encoding')

    def blosum(self, seq):
        return self.__encoding__(seq, self.Residue.getBlosumDic(), 'Blosum Encoding')
