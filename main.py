from Training import *
from Application import *
from explanation import *


if __name__ == '__main__':
    print('\nWe are running main.py!\nGood Luck!\n')
    parser = argparse.ArgumentParser(description='DeepMEP')
    parser.add_argument('--disease_i', type=int, default=None, help='PTM Enrichment')
    parser.add_argument('--data_from', type=str, default=None, help='PTM Enrichment')
    # parser.add_argument('--variant_type', type=str, default=None, help='PTM Enrichment')
    # parser.add_argument('--q', type=float, default=None, help='PTM Enrichment')
    args = parser.parse_args()
    disease_i = args.disease_i
    data_from = args.data_from
    # variant_type = args.variant_type
    # q = args.q
    # print((disease_i, data_from))

    # ProteinSequenceAnnotation('uniprot').split_dataset_at_protein_level()

    for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        enricher = EnrichAnalyzer(data_from=data_from, variant_type='common', ptm_sig_threshold_q=q, sample_type='random', full_set='Missense', repeat_times=1000)
        enricher.single_disease_ptm_enrichment(disease_i)

    # SNPAnnotation().map_eqtl_snp_id()

    # Applier(data_from='uniprot', device_id=7).apply_model()
    # Applier(data_from='epsd_family', device_id=7).apply_model()
    # Applier(data_from='cbmap', device_id=7).apply_model()

    # Train intensity model
    # search_rs = []
    # for embedding_layer in range(1, 7):
    #     for window_size in range(5, 17):
    #         print(f"window_size: {window_size}; embedding_layer: {embedding_layer};")
    #         trainer = Trainer(
    #             window_size=window_size, multi_sites_joint=False, lr=0.0006, weight_decay=0.0008, batch_size=2048,
    #             epochs=20, data_from='cbmap', encode_mode='esm-embedding', device=7,
    #             mlp_feature_num=12, mlp_mediate_neuron_num=512, mlp_drop_p=[0.5, 0.5, 0.15],
    #             embedding_layer_number=embedding_layer, embedding_model_version=320, is_print=False
    #         )
    #         r2 = trainer.train_model()[0][0]
    #         search_rs.append((window_size, embedding_layer, r2))
    # pd.DataFrame(search_rs, columns=['window_size', 'embedding_layer', 'r2']).to_csv(
    #     'rs/our_model/uniprot/search_rs_cbmap.txt', sep='\t', index=False
    # )

    # Train uniprot model
    # search_rs = []
    # for embedding_layer in range(30, 31):
    #     for window_size in range(16, 17):
    #         print(f"window_size: {window_size}; embedding_layer: {embedding_layer};")
    #         trainer = Trainer(
    #             window_size=window_size, lr=0.0001, weight_decay=0.0001, batch_size=5120,
    #             epochs=10, data_from='uniprot', encode_mode='esm-embedding', device=7,
    #             mlp_feature_num=12, mlp_mediate_neuron_num=512, alpha=0.03, mlp_drop_p=[0.5, 0.15, 0.1],
    #             embedding_layer_number=embedding_layer, embedding_model_version=640, is_print=True,
    #             task_type='prediction'
    #         )
    #         _, sen, roc_auc, _, _, pr_auc = trainer.train_model()[0]
    #         search_rs.append((window_size, embedding_layer, sen, pr_auc, roc_auc))
    # pd.DataFrame(search_rs, columns=['window_size', 'embedding_layer', 'sen', 'pr_auc', 'roc_auc']).to_csv(
    #     'rs/our_model/uniprot/search_rs.txt', sep='\t', index=False
    # )

    # train epsd model --group
    # alpha = (torch.tensor([0.0, 0.07, 0.12, 0.11, 0.15, 0.05, 0.14, 0.14, 0.10, 0.1])).tolist()
    # Trainer(
    #     window_size=15, lr=0.001, weight_decay=0.001, batch_size=256,
    #     epochs=50, data_from='epsd_group', encode_mode='esm-embedding', device=7,
    #     mlp_feature_num=12, mlp_mediate_neuron_num=512, mlp_drop_p=[0.5, 0.5, 0.15],
    #     embedding_layer_number=2, embedding_model_version=320, alpha=0
    # ).train_model()

    # train epsd model --family
    # alpha = (torch.tensor([0.0, 0.07, 0.12, 0.11, 0.15, 0.05, 0.14, 0.14, 0.10, 0.1])).tolist()
    # for embedding_layer in range(1, 7):
    #     for window_size in range(5, 17):
    #         print(f"window_size: {window_size}; embedding_layer: {embedding_layer};")
    #         Trainer(
    #             window_size=window_size, lr=0.001, weight_decay=0.001, batch_size=256,
    #             epochs=30, data_from='epsd_family', encode_mode='esm-embedding', device=7,
    #             mlp_feature_num=12, mlp_mediate_neuron_num=512, mlp_drop_p=[0.5, 0.5, 0.15], is_print=False,
    #             embedding_layer_number=embedding_layer, embedding_model_version=320, alpha=0.
    #         ).train_model()

    # train interpretable model for CBMAP
    # Trainer(
    #     window_size=5, data_from='cbmap', encode_mode='one-hot', batch_size=2048, epochs=20,
    #     lr=0.0006, weight_decay=0.0008, conv_out_channels=20, device=7, task_type='explanation'
    # ).train_model()

    # datasetor = TrainingDataset(window_size=16, use_structure=False, data_from='cbmap')
    # datasetor.getSeqInstance()
    # datasetor.getDataset(encode_mode='esm-embedding')

    # application_dataset = ApplicationDataset(50, 7)
    # application_dataset.getApplicationInstances()

    # Genome2Proteome()
