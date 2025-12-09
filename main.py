from Training import *
from Application import *
from explanation import *


if __name__ == '__main__':
    # /Users/yuexu/Documents/PyWork/DeepMEP/
    print('\nWe are running main.py!\nGood Luck!\n')
    # parser = argparse.ArgumentParser(description='DeepMEP')
    # ========================================================================================
    # ============================variant-level Enrichment analysis===========================
    # parser.add_argument('--disease_i', type=int, default=None, help='DeepMEP Enrichment')
    # parser.add_argument('--data_from', type=str, default=None, help='DeepMEP Enrichment')
    # parser.add_argument('--variant_type', type=str, default=None, help='DeepMEP Enrichment')
    # parser.add_argument('--q', type=float, default=None, help='DeepMEP Enrichment')
    # parser.add_argument('--task_i', type=int, default=None, help='DeepMEP Enrichment')
    # args = parser.parse_args()
    # tissue_i = args.tissue_i
    # data_from = args.data_from
    # variant_type = args.variant_type
    # q = args.q
    # disease_i = args.disease_i
    # data_from, variant_type, full_set, sample_type = [
    #     ('uniprot', 'common', 'Missense', 'distribution'),
    #     ('cbmap', 'common', 'Missense', 'distribution'),
    #     ('uniprot', 'common', 'Missense', 'random'),
    #     ('cbmap', 'common', 'Missense', 'random'),
    #     ('uniprot', 'common', 'ALL', 'distribution'),
    #     ('cbmap', 'common', 'ALL', 'distribution'),
    #     ('uniprot', 'rare', 'Missense', 'random'),
    #     ('cbmap', 'rare', 'Missense', 'random'),
    # ][args.task_i]
    # q_lst = [[-0.5, 1.0], [-0.5]][args.task_i]
    # q_lst = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # q_lst = [0.5] if variant_type == 'rare' else q_lst
    # for q in q_lst:
    #     enricher = EnrichAnalyzer(
    #         data_from=data_from, variant_type=variant_type, ptm_sig_threshold_q=q, element_type='variant',
    #         sample_type=sample_type, full_set=full_set, repeat_times=1000
    #     )
    #     enricher.PTMEnrichment()
    # data_from: Any
    # for data_from in ['uniprot']:
    #     for q in [0.5]:
    #         enricher = EnrichAnalyzer(
    #             data_from=data_from, variant_type='rare', ptm_sig_threshold_q=q, element_type='variant',
    #             sample_type='random', full_set='Missense', repeat_times=1000, phos_effect='max'
    #         )
    #         enricher.single_disease_ptm_enrichment(disease_ii=disease_i)

    # ============================gene-level Enrichment analysis===========================
    # parser.add_argument('--tissue_i', type=int, default=None, help='Run PrediXcan')
    # for i in range(539):
    #     enricher.single_disease_ptm_enrichment(i)



    # ============================Model Training===========================================
    # =====================================================================================
    # ============================Train intensity model====================================
    # search_rs = []
    # for fold_i in range(5):
    #     for embedding_layer in range(1, 7):
    #         for window_size in range(12, 13):
    #             print(f"CBMAP: fold_i: {fold_i}; window_size: {window_size}; embedding_layer: {embedding_layer};")
    #             trainer = Trainer(
    #                 window_size=window_size, lr=0.0006, weight_decay=0.0008, batch_size=2048,
    #                 epochs=20, data_from='cbmap', encode_mode='esm-embedding', device=7,
    #                 mlp_feature_num=12, mlp_mediate_neuron_num=512, mlp_drop_p=[0.5, 0.5, 0.15],
    #                 embedding_layer_number=embedding_layer, embedding_model_version=320, is_print=False,
    #                 fold_i=fold_i, task_type='prediction'
    #             )
    #             r2 = trainer.train_model()[0][0]
    #         search_rs.append((window_size, embedding_layer, r2))
    # pd.DataFrame(search_rs, columns=['window_size', 'embedding_layer', 'r2']).to_csv(
    #     'rs/our_model/uniprot/search_rs_cbmap.txt', sep='\t', index=False
    # )

    # ============================Train uniprot model======================================
    # search_rs = []
    # for fold_i in range(5):
    #     for embedding_layer in range(1, 7):
    #         for window_size in range(15, 16):
    #             print(f"Uniprot: fold_i: {fold_i}; window_size: {window_size}; embedding_layer: {embedding_layer};")
    #             trainer = Trainer(
    #                 window_size=window_size, lr=0.0001, weight_decay=0.0001, batch_size=5120,
    #                 epochs=10, data_from='uniprot', encode_mode='esm-embedding', device=7,
    #                 mlp_feature_num=12, mlp_mediate_neuron_num=512, alpha=0.03, mlp_drop_p=[0.5, 0.15, 0.1],
    #                 embedding_layer_number=embedding_layer, embedding_model_version=320, is_print=False,
    #                 task_type='prediction', fold_i=fold_i
    #             )
    #             _, sen, roc_auc, _, _, pr_auc = trainer.train_model()[0]
    #         search_rs.append((window_size, embedding_layer, sen, pr_auc, roc_auc))
    # pd.DataFrame(search_rs, columns=['window_size', 'embedding_layer', 'sen', 'pr_auc', 'roc_auc']).to_csv(
    #     'rs/our_model/uniprot/search_rs.txt', sep='\t', index=False
    # )

    # ============================train epsd group model=================================
    # alpha = (torch.tensor([0.0, 0.07, 0.12, 0.11, 0.15, 0.05, 0.14, 0.14, 0.10, 0.1])).tolist()
    # Trainer(
    #     window_size=15, lr=0.001, weight_decay=0.001, batch_size=256,
    #     epochs=50, data_from='epsd_group', encode_mode='esm-embedding', device=7,
    #     mlp_feature_num=12, mlp_mediate_neuron_num=512, mlp_drop_p=[0.5, 0.5, 0.15],
    #     embedding_layer_number=2, embedding_model_version=320, alpha=0
    # ).train_model()

    # ============================train epsd family model=================================
    # alpha = (torch.tensor([0.0, 0.07, 0.12, 0.11, 0.15, 0.05, 0.14, 0.14, 0.10, 0.1])).tolist()
    # for fold_i in range(5):
    #     for embedding_layer in range(1, 7):
    #         for window_size in range(7, 8):
    #             print(f"EPSD family: fold_i: {fold_i}; window_size: {window_size}; embedding_layer: {embedding_layer};")
    #             Trainer(
    #                 window_size=window_size, lr=0.001, weight_decay=0.001, batch_size=256,
    #                 epochs=30, data_from='epsd_family', encode_mode='esm-embedding', device=7,
    #                 mlp_feature_num=12, mlp_mediate_neuron_num=512, mlp_drop_p=[0.5, 0.5, 0.15], is_print=False,
    #                 embedding_layer_number=embedding_layer, embedding_model_version=320, alpha=0.,
    #                 task_type='prediction', fold_i=fold_i
    #             ).train_model()

    # =========================train interpretable model for CBMAP========================
    # Trainer(
    #     window_size=5, data_from='cbmap', encode_mode='one-hot', batch_size=2048, epochs=20,
    #     lr=0.0006, weight_decay=0.0008, conv_out_channels=20, device=7, task_type='explanation'
    # ).train_model()

    # ============================MusiteDeep==============================================
    # from existing_tools.MusiteDeep import *
    # trainMusiteDeep(device_id=2, model_type='general', cross_validation=True)
    # evaluate_MusiteDeep(cross_validation=True)

    # ============================DeepPhos===============================================
    # from existing_tools.DeepPhos import *
    # trainDeepPhos(device_id=[0, 1], model_type='intensity', cross_validation=True)
    # evaluate_DeepPhos()

    # ============================ EMBER ================================================
    # from existing_tools.EMBER import *
    # train_siamese_embedder(device_id=2, class_name='group')
    # embedding_umap_projection(embedding_device=0)
    # train_ember(device_id=2, class_name='group')
    # get_siamese_motifs()

    # ============================Phosformer ===========================================
    # from existing_tools.phosformer import *
    # evaluate_Phosformer()

    # ============================DeepPPSite ===========================================
    # from existing_tools.DeepPPSite import *
    # residue: Any
    # data_from: Any
    # for residue in ['S', 'T', 'Y']:
    #     for data_from in ['uniprot', 'cbmap']:
    #         train_DeepPPSite(residue, data_from, 0, True)
    # evaluate_DeepPPSite()

    # ============================Other Code ===========================================
    # runPrediXcan(48, tissue_i)

    # ProteinStructureFeature().compute_all_sites_structure()

    # ProteinSequenceAnnotation('uniprot').split_dataset_at_protein_level()

    # AbstractFeatureVision(7).plot_figure(15, 10, 0., 0.)

    # GWASProcessor().process_New202_batch2()

    # SNPAnnotation().map_eqtl_snp_id()

    # datasetor = TrainingDataset(window_size=7, use_structure=False, data_from='epsd_family')
    # datasetor.getSeqInstance()
    # datasetor.getDataset(encode_mode='esm-embedding', para_i=disease_i)

    # application_dataset = ApplicationDataset(50, 7)
    # application_dataset.getApplicationInstances()

    # Genome2Proteome()

    # from Protein import PDB2DF
    # PDB2DF()
