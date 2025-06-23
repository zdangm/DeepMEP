from Protein import Embedding
from Model import *
from DataProcess import *


class Trainer:
    def __init__(self,
                 window_size: int,  # the size of the upstream and downstream of the focused site
                 data_from: Literal['uniprot', 'epsd_group', 'epsd_family', 'cbmap'],  # 'private' | 'uniprot', the source of data
                 encode_mode: Literal['one-hot', 'esm-embedding'],  # 'one-hot' | 'embedding', the type of encoding the protein sequences
                 batch_size: int,  # the size of mini-batch learning
                 epochs: int,  # the times of going through the dataset
                 lr: float,  # learning rate
                 weight_decay: float,  # weight of l2 refularization
                 task_type: Literal['prediction', 'explanation'],
                 mlp_mediate_neuron_num: Optional[int] = None,  # the number of intermediate neurons in the output block
                 mlp_feature_num: Optional[int] = None,  # the number of neurons for the out neuron
                 mlp_drop_p: Optional[List[float]] = None,  # the probability in dropout layer
                 conv_out_channels: Optional[int] = None,  # the out channels in explanable model
                 embedding_layer_number: Optional[int] = None,  # the No. of layer to return in ESM2
                 embedding_model_version: Literal[320, 640, 1280, 2560, 5120] = 320,  # the embedding dimension of different versions of models
                 alpha: Optional[Union[float, List]] = None,  # the class weight of Focal Loss that deals with the class imbalance
                 gamma: float = 2.0,  # the dynamic scaling factor of Focal Loss that control the contribution of difficult examples
                 pool_size: int = 1,  # esm embedding max pool size
                 device: Optional[Union[str, int, List[int]]] = None,  # 'cpu' | int | list, devices for training
                 is_print: bool = True,  # whether printing the metrics of each epoch
                 is_save: bool = True,  # whether saving the training results
                 ):
        # get the number of input channels
        in_channels = 0
        if encode_mode == 'one-hot':
            in_channels = 20
        elif encode_mode == 'esm-embedding':
            in_channels = int(embedding_model_version / pool_size)

        # get the number and name of output labels
        labels_num = 0
        if data_from == 'uniprot':
            labels_num = 2
            self.label_name = ['Phosphorylation']
            model_type = 'classification'
        elif data_from == 'epsd_group':
            self.label_name = ProteinSequenceAnnotation.get_kinase_group()
            labels_num = len(self.label_name) + 1
            model_type = 'classification'
        elif data_from == 'epsd_family':
            self.label_name = ProteinSequenceAnnotation.get_kinase_family()
            labels_num = len(self.label_name) + 1
            model_type = 'classification'
        elif data_from == 'cbmap':
            labels_num = 1
            self.label_name = ['Intensity']

        model_type: Any = 'regression' if data_from == 'cbmap' else 'classification'

        # generate metrics lst
        if data_from != 'cbmap':
            self.metrics_name_lst = [
                'train_acc', 'train_sen', 'train_roc_auc', 'train_f1', 'train_precision', 'train_pr_auc',
                'val_acc  ', 'val_sen  ', 'val_roc_auc  ', 'val_f1  ', 'val_precision  ', 'val_pr_auc  ',
            ]
        else:
            self.metrics_name_lst = ['train_R2', 'val_R2']
        basic_metrics = {metric_name: [] for metric_name in self.metrics_name_lst}
        label_metrics_dict: Any
        label_metrics_dict = {name: basic_metrics for name in self.label_name}
        label_metrics_dict['train_loss'] = []
        label_metrics_dict['val_loss'] = []
        self.metrics = label_metrics_dict

        # set criteria
        if data_from == 'cbmap':
            # self.criteria = nn.MSELoss()
            self.criteria = nn.HuberLoss()
        elif data_from == 'epsd_group' or data_from == 'epsd_family':
            self.criteria = FocalLoss(alpha+alpha, gamma, labels_num*2, True)
        elif data_from == 'uniprot':
            self.criteria = FocalLoss(alpha, gamma, labels_num, True)

        # initialize the model
        if task_type == 'prediction':
            model = BasicPredictor(
                conv_in_channels=in_channels, seq_len=2 * window_size + 1, labels_num=labels_num,
                device_ids=device, mlp_feature_num=mlp_feature_num, drop_p=mlp_drop_p,
                mlp_mediate_neuron_num=mlp_mediate_neuron_num, model_type=model_type
            )
        else:
            model = InterpretableModel(
                seq_len=2*window_size + 1, conv_in_channels=in_channels,
                conv_out_channels=conv_out_channels, model_type=model_type, device_ids=device
            )
        print(model)
        self.main_device = get_main_device(device)
        self.model, self.best_model = model, model

        # global parameters
        self.window_size = window_size
        self.is_print, self.is_save = is_print, is_save
        self.batch_size, self.epoches = batch_size, epochs
        self.data_from = data_from
        self.task_type = task_type

        # embedder
        self.encode_mode = encode_mode
        embedding_device_id = device[0] if isinstance(device, list) else device
        if encode_mode == 'esm-embedding':
            self.esm_embedder = Embedding(device_id=embedding_device_id, pool_size=pool_size,
                                          model_version=embedding_model_version)
            self.return_layer = embedding_layer_number

        # Adam opimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # get dataset
        Dataseter = TrainingDataset(window_size=window_size, use_structure=False, data_from=data_from)
        self.dataset_dic = Dataseter.getDataset(encode_mode=encode_mode, require_test=False)

    def train_model(self):
        if self.encode_mode == 'esm-embedding':
            return self.train_esm_embedding_model()
        elif self.encode_mode == 'one-hot':
            return self.train_onehot_model()
        return None

    def _train_model_(self, batch_i: int, batch_x: torch.Tensor, batch_y: torch.Tensor, batch_mask: torch.Tensor):
        batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)
        batch_mask = batch_mask.to(self.main_device)
        self.model.train()
        pred = self.model(batch_x)
        loss = self.criteria(pred.flatten(1, -1), batch_y.flatten(1, -1))
        if self.data_from == 'epsd_family' or self.data_from == 'epsd_group':
            pred, batch_y = pred[:, :, 1], batch_y[:, :, 1]
        batch_metrics_lst, _ = self.compute_model_metrics(preds=pred, labels=batch_y, set_type='batch')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy(), batch_metrics_lst

    def _evaluate_val_(self, val_x, val_y, val_mask):
        val_pred = self.model.predict(val_x).cpu()
        val_loss = self.criteria(val_pred.flatten(1, -1), val_y.flatten(1, -1))
        if self.data_from == 'epsd_family' or self.data_from == 'epsd_group':
            val_pred, val_y = val_pred[:, :, 1], val_y[:, :, 1]
        val_metrics_lst, val_total = self.compute_model_metrics(preds=val_pred, labels=val_y, set_type='val')
        return val_metrics_lst, val_total, val_loss.cpu().detach().numpy()

    def train_esm_embedding_model(self):
        best_val_total, best_model_wts, best_val_metrics_lst = 0, None, None
        train_y, train_seq, train_mask, _, _, _, val_y, val_seq, val_mask = self.dataset_dic['context']
        val_x = self.esm_embedder.bulk_embedding(val_seq, self.return_layer).transpose(1, 2).to(self.main_device)
        if self.data_from == 'cbmap':
            train_y, val_y = train_y[:, self.window_size].reshape(-1, 1), val_y[:, self.window_size].reshape(-1, 1)
            # train_mask, val_mask = (train_y > -999).int().float(), (val_y > -999).int().float()
        elif self.data_from == 'epsd_family' or self.data_from == 'epsd_group':
            train_pos = train_y[:, self.window_size, 0] == 0
            train_y, train_mask = train_y[train_pos], train_mask[train_pos]
            train_seq = [train_seq[i] for i, b in enumerate(train_pos.tolist()) if b]
            train_y, val_y = train_y[:, self.window_size, :], val_y[:, self.window_size, :]
            # batch, classes -> batch, classes, 2
            train_y = torch.stack([(train_y + 1) % 2, train_y], dim=-1)
            val_y = torch.stack([(val_y + 1) % 2, val_y], dim=-1)
        else:
            train_y, val_y = train_y[:, self.window_size, :], val_y[:, self.window_size, :]

        for epoch in range(self.epoches):
            data_loader = load_str_batch(batch_size=self.batch_size, feature=train_seq,
                                         label=train_y, label_mask=train_mask)
            for batch_i, (batch_seq, batch_y, batch_mask, _) in enumerate(data_loader):
                batch_x = self.esm_embedder.bulk_embedding(batch_seq, self.return_layer).transpose(1, 2)
                batch_loss, batch_metrics_lst = self._train_model_(batch_i, batch_x, batch_y, batch_mask)
                val_metrics_lst, val_total, val_loss = self._evaluate_val_(val_x, val_y, val_mask)
                if val_total > best_val_total:
                    best_val_total = val_total
                    best_val_metrics_lst = val_metrics_lst
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                self.updateMetrics(epoch=epoch, batch_i=batch_i, best_val=best_val_total,
                                   train_loss=batch_loss, val_loss=val_loss,
                                   batch_metrics_lst=batch_metrics_lst, val_metrics_lst=val_metrics_lst)
        if best_model_wts is None:
            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_val_metrics_lst = [(0,)]
        self.best_model.load_state_dict(best_model_wts)
        if self.is_print:
            if self.data_from == 'cbmap':
                print(f'best_val_r2: {best_val_metrics_lst[0][0]:.3f}')
            else:
                print(f'best_val_metrics: {best_val_metrics_lst}')
        self.save_rs()
        return best_val_metrics_lst

    def train_onehot_model(self):
        best_val_total, best_model_wts, best_val_metrics = 0, None, None
        train_y, train_x, train_mask, _, _, _, val_y, val_x, val_mask = self.dataset_dic['context']
        if self.data_from == 'cbmap':
            train_y, val_y = train_y[:, self.window_size].reshape(-1, 1), val_y[:, self.window_size].reshape(-1, 1)
        val_x = val_x.to(self.main_device)
        dataset = torch.utils.data.TensorDataset(train_x, train_y, train_mask)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        for epoch in range(self.epoches):
            for batch_i, (batch_x, batch_y, batch_mask) in enumerate(data_loader):
                batch_loss, batch_metrics_lst = self._train_model_(batch_i, batch_x, batch_y, batch_mask)
                val_metrics_lst, val_total, val_loss = self._evaluate_val_(val_x, val_y, val_mask)
                if val_total > best_val_total:
                    best_val_total = val_total
                    best_val_metrics = val_metrics_lst
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                self.updateMetrics(epoch=epoch, batch_i=batch_i, best_val=best_val_total,
                                   train_loss=batch_loss, val_loss=val_loss,
                                   batch_metrics_lst=batch_metrics_lst, val_metrics_lst=val_metrics_lst)
        if best_model_wts is None:
            best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_model.load_state_dict(best_model_wts)
        print(best_val_metrics)
        self.save_rs()

    def train_bootstrap_esm_model(self):
        best_val_total, best_model_wts, best_val_metrics = 0, None, None
        train_y, train_seq, train_mask, _, _, _, val_y, val_seq, val_mask = self.dataset_dic['context']
        val_x = self.esm_embedder.bulk_embedding(val_seq, self.return_layer).transpose(1, 2).cpu()
        train_all_i = torch.arange(0, train_y.shape[0])
        train_pos_i = train_all_i[train_y[:, self.window_size, 0] == 0]
        train_neg_i = train_all_i[train_y[:, self.window_size, 0] == 1]
        train_y_pos, train_y_neg = train_y[train_pos_i], train_y[train_neg_i]
        train_mask_pos, train_mask_neg = train_mask[train_pos_i], train_mask[train_neg_i]
        train_seq_pos, train_seq_neg = [train_seq[i] for i in train_pos_i], [train_seq[i] for i in train_neg_i]
        pos_len = len(train_seq_pos)
        bootstrap_i = math.ceil(len(train_seq_neg) / pos_len)
        for epoch in range(self.epoches):
            batch_loss, batch_i, batch_metrics_lst = 0, 0, None
            for t in range(bootstrap_i):
                select_train_seq_neg = train_seq_neg[t*pos_len: (t+1)*pos_len]
                select_train_y_neg = train_y_neg[t*pos_len: (t+1)*pos_len]
                select_train_mask_neg = train_mask_neg[t*pos_len: (t+1)*pos_len]
                select_train_seq = train_seq_pos + select_train_seq_neg
                select_train_y = torch.cat([train_y_pos, select_train_y_neg], dim=0)
                select_train_mask = torch.cat([train_mask_pos, select_train_mask_neg], dim=0)
                data_loader = load_str_batch(batch_size=self.batch_size, feature=select_train_seq,
                                             label=select_train_y, label_mask=select_train_mask)
                for batch_i, (batch_seq, batch_y, batch_mask, _) in enumerate(data_loader):
                    batch_x = self.esm_embedder.bulk_embedding(batch_seq, self.return_layer).transpose(1, 2)
                    batch_loss, batch_metrics_lst = self._train_model_(batch_i, batch_x, batch_y, batch_mask)
                    val_metrics_lst, val_total, val_loss = self._evaluate_val_(val_x, val_y, val_mask)
                    if val_total > best_val_total:
                        best_val_total = val_total
                        best_val_metrics = val_metrics_lst
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.updateMetrics(epoch=epoch, batch_i=batch_i, best_val=best_val_total,
                                       train_loss=batch_loss, val_loss=val_loss,
                                       batch_metrics_lst=batch_metrics_lst, val_metrics_lst=val_metrics_lst)
        self.best_model.load_state_dict(best_model_wts)
        self.save_rs()

    def compute_model_metrics(self, preds, labels, set_type):
        metrics_lst = []
        total_metrics = 0
        if self.data_from == 'cbmap':
            for i in range(len(self.label_name)):
                r2 = metrics.r2_score(y_true=labels.detach().cpu().numpy(),
                                      y_pred=preds.detach().cpu().numpy())
                total_metrics += r2
                metrics_lst.append((r2,))
        else:
            for i in range(len(self.label_name)):
                acc, sen, roc_auc, f1, precision, pr_auc = compute_metrics(
                    preds=preds[:, i + 1], targets=labels[:, i + 1]
                )
                total_metrics += roc_auc
                metrics_lst.append((acc, sen, roc_auc, f1, precision, pr_auc))
        return metrics_lst, total_metrics / len(self.label_name)

    def updateMetrics(self, epoch, batch_i, best_val, train_loss, val_loss, batch_metrics_lst, val_metrics_lst):
        metrics_dict = {}
        for i, label_name in enumerate(self.label_name):
            label_metrics_value_lst = list(batch_metrics_lst[i]) + list(val_metrics_lst[i])
            metrics_dict[label_name] = {
                self.metrics_name_lst[j]: label_metrics_value_lst[j]
                for j in range(len(self.metrics_name_lst))
            }
        if self.is_print:
            print(f'epoch: {epoch}, batch_i: {batch_i}, best_val: {best_val:.3f}, '
                  f'train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}')
        for label_name in self.label_name:
            label_metrics_dict = metrics_dict[label_name]
            label_train_sentence, label_val_sentence, i = f'{label_name:<10}: ', f'{label_name:<10}: ', 0
            for metrics_name, metrics_value in label_metrics_dict.items():
                self.metrics[label_name][metrics_name].append(metrics_value)
                if i < len(self.metrics_name_lst) / 2:
                    label_train_sentence = f'{label_train_sentence}{metrics_name}: {metrics_value:.3f}, '
                else:
                    label_val_sentence = f'{label_val_sentence}{metrics_name}: {metrics_value:.3f}, '
                i += 1
            print(f'{label_train_sentence}\n{label_val_sentence}') if self.is_print else None

        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)

    def plot_cbmap_val_pred(self, filename: str):
        _, _, _, _, _, _, val_y, val_seq, _ = self.dataset_dic['context']
        if self.encode_mode == 'esm-embedding':
            val_x = self.esm_embedder.bulk_embedding(val_seq, self.return_layer).transpose(1, 2).to(self.main_device)
        else:
            val_x = val_seq.to(self.main_device)
        val_y = val_y[:, self.window_size].detach().cpu().numpy()
        val_y_pred = self.best_model.predict(val_x).cpu().detach().numpy().reshape(-1)
        # get figure 1 plot_data
        pos_threshold = np.quantile(val_y, 0.55)
        tp_bool = (val_y > pos_threshold) & (val_y_pred > pos_threshold)
        tn_bool = (val_y <= pos_threshold) & (val_y_pred <= pos_threshold)
        fp_bool = (val_y <= pos_threshold) & (val_y_pred > pos_threshold)
        fn_bool = (val_y > pos_threshold) & (val_y_pred <= pos_threshold)
        tp_y, tp_y_pred = val_y[tp_bool], val_y_pred[tp_bool]
        tn_y, tn_y_pred = val_y[tn_bool], val_y_pred[tn_bool]
        fp_y, fp_y_pred = val_y[fp_bool], val_y_pred[fp_bool]
        fn_y, fn_y_pred = val_y[fn_bool], val_y_pred[fn_bool]
        r2 = metrics.r2_score(y_true=val_y, y_pred=val_y_pred)

        # tune the positive parameters
        quantiles = np.linspace(0.05, 0.95, 19)
        pos_threshold_lst = np.quantile(val_y, quantiles)
        roc_auc_lst, pr_auc_lst = [], []
        for q, r in zip(quantiles, pos_threshold_lst):
            val_label = (val_y > r).astype(int)
            roc_auc_lst.append(metrics.roc_auc_score(y_true=val_label, y_score=val_y_pred))
            pred_label = (val_y_pred > r).astype(int)
            precision_lst, recall_lst, _ = metrics.precision_recall_curve(val_label, pred_label)
            pr_auc = metrics.auc(recall_lst, precision_lst)
            pr_auc_lst.append(pr_auc)

        plt.figure(figsize=(12, 6), facecolor='white')
        plt.subplot(1, 2, 1)
        # plt.scatter(x=val_y, y=val_y_pred, label=f'R-square = {r2:.3f}', s=.5)
        plt.axhline(y=pos_threshold, linestyle='--', color='black', label='Positive Threshold')
        plt.axvline(x=pos_threshold, linestyle='--', color='black')
        plt.axline((0, 0), slope=1, linestyle='--', color='red', label='y = x')
        plt.scatter(x=tp_y, y=tp_y_pred, label = f'TP: {np.sum(tp_bool)}', s=.6)
        plt.scatter(x=tn_y, y=tn_y_pred, label=f'TN: {np.sum(tn_bool)}', s=.6)
        plt.scatter(x=fp_y, y=fp_y_pred, label=f'FP: {np.sum(fp_bool)}', s=.6)
        plt.scatter(x=fn_y, y=fn_y_pred, label=f'FN: {np.sum(fn_bool)}', s=.6)
        plt.title(f'Prediction on val-set (R2={r2:.3f})')
        plt.xlabel('True Intensity')
        plt.ylabel('Predicted Intensity')
        plt.legend(markerscale=5)
        plt.axis('equal')

        plt.subplot(1, 2, 2)
        plt.plot(quantiles, np.array(roc_auc_lst), marker='s', label=f'Best ROC_AUC: {max(roc_auc_lst):.3f}')
        plt.plot(quantiles, np.array(pr_auc_lst), marker='s', label=f'Best PR_AUC: {max(pr_auc_lst):.3f}')
        plt.title('Metrics under different thresholds')
        plt.xlabel('Quantiles')
        plt.ylabel('ROC AUC')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_model_param_dist(model: ModelInterface, filename: str):
        model_param = model.state_dict()
        param_names = []
        for param in model_param.keys():
            if param.split('.')[-1] == 'bias':
                continue
            if '.bn' in param:
                continue
            if 'num_batches_tracked' in param:
                continue
            if 'running_' in param:
                continue
            param_names.append(param)
        nrow, ncol = math.ceil(len(param_names) / 4), 4
        plt.figure(figsize=(ncol * 6, nrow * 6), facecolor='white')
        for i, param_name in enumerate(param_names):
            plt.subplot(nrow, ncol, i+1)
            sns.histplot(
                model_param[param_name].detach().cpu().numpy().flatten(),
                bins=30, kde=True, color='green', edgecolor='black'
            )
            plt.title(f"{param_name.replace('.weight', '')}")
            plt.xlabel('Weight parameter')
            plt.ylabel('Density')
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        plt.savefig(filename)
        plt.close()

    def plotTrainingMetrics(self, filename: str):
        if self.data_from == 'epsd_family' or self.data_from == 'epsd_group':
            nrow = 1
        elif self.data_from == 'cbmap':
            nrow = 2
        else:
            nrow = 3

        plt.figure(figsize=(8, nrow*5), facecolor='white')

        plt.subplot(nrow, 1, 1)
        plt.plot(range(len(self.metrics['train_loss'])), self.metrics['train_loss'], label='train loss')
        plt.plot(range(len(self.metrics['val_loss'])), self.metrics['val_loss'], label='val loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        if nrow >= 2:
            plt.subplot(nrow, 1, 2)
            if self.data_from == 'cbmap':
                plt.plot(range(len(self.metrics['Intensity']['train_R2'])),
                         self.metrics['Intensity']['train_R2'], label=f'train r-square')
                plt.plot(range(len(self.metrics['Intensity']['val_R2'])),
                         self.metrics['Intensity']['val_R2'], label=f'val r-square')
                plt.axhline(y=max(self.metrics['Intensity']['val_R2']), linestyle='--', color='black')
                plt.title('R-square')
            else:
                for label_name in self.label_name:
                    plt.plot(range(len(self.metrics[label_name]['val_sen  '])),
                             self.metrics[label_name]['val_sen  '], label=f'{label_name} val sen')
                    plt.plot(range(len(self.metrics[label_name]['val_precision  '])),
                             self.metrics[label_name]['val_precision  '], label=f'{label_name} val precision')
                plt.title('Accuracy, Sensitivity, Precision')
            plt.xlabel('Iterations')
            plt.ylabel('Metrics')
            plt.legend()

        if nrow >= 3:
            plt.subplot(nrow, 1, 3)
            for label_name in self.label_name:
                plt.plot(range(len(self.metrics[label_name]['train_roc_auc'])),
                         self.metrics[label_name]['train_roc_auc'], label=f'{label_name} train')
                plt.plot(range(len(self.metrics[label_name]['val_roc_auc  '])),
                         self.metrics[label_name]['val_roc_auc  '], label=f'{label_name} val')
                plt.axhline(y=max(self.metrics[label_name]['val_roc_auc  ']), linestyle='--', color='black',
                            label=f'{label_name}_{max(self.metrics[label_name]["val_roc_auc  "]):.3f}')
            plt.title('ROC AUC')
            plt.xlabel('Iterations')
            plt.ylabel('Metrics')
            plt.legend()
        plt.savefig(filename)
        plt.close()

    def save_rs(self):
        if self.is_save:
            file_dir = f'rs/our_model/{self.data_from}'
            make_dir(file_dir)
            file_dir = f'{file_dir}/{self.task_type}_{self.encode_mode}'
            make_dir(file_dir)
            # make save directory
            file_dir = f'{file_dir}/{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
            make_dir(file_dir)
            print(file_dir)
            # save model
            if self.encode_mode == 'one-hot':
                best_model_path = f'{file_dir}/model_{self.window_size}.pth'
            else:
                best_model_path = f'{file_dir}/model_{self.window_size}_{self.return_layer}.pth'
            Saver(variable=self.best_model, filename=best_model_path).save_pth()
            # save running metrics
            self.plotTrainingMetrics(filename=f'{file_dir}/Metrics.pdf')
            # save val pred
            if self.data_from == 'cbmap':
                self.plot_cbmap_val_pred(filename=f'{file_dir}/val_pred_scatter.pdf')
            # save metrics
            Saver(variable=self.metrics, filename=f'{file_dir}/metrics.pkl').save_pkl()
            # save parameters' distribution of model
            self.plot_model_param_dist(filename=f'{file_dir}/model_params_dist.pdf', model=self.best_model)
            # save gradient of model
            # runner.plotModelGrad(filename=f'{file_dir}/Gradient.pdf')
