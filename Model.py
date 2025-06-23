from utils import *


class ModelInterface(nn.Module, ABC):
    def __init__(self, device_ids):
        super(ModelInterface, self).__init__()
        judging = device_ids is None or device_ids == 'cpu' or isinstance(device_ids, int)
        self.parallel = False if judging else True

    @abstractmethod
    def to_device(self):
        pass

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x) -> torch.Tensor:
        pass


class BasicPredictor(ModelInterface):
    def __init__(self,
                 seq_len: int,  # the length of input sequence
                 conv_in_channels: int,  # the number of channels of input data
                 mlp_mediate_neuron_num: int,  # the number of intermediate neurons in the output block
                 mlp_feature_num: int,  # the number of neurons for the out neuron
                 labels_num: int,  # the number of neurons in the output layer
                 drop_p: List[float],  # the probability in dropout layer
                 slope: float = 0.01,  # the slope of leaky relu activation function
                 return_last_layer: bool = True,  # whether returning the output of last layer as final output
                 device_ids: Optional[Union[List, int, str]] = None,  # the device used in model
                 model_type: Literal['regression', 'classification'] = 'regression'  # the type of model
                 ):
        super(BasicPredictor, self).__init__(device_ids)
        self.return_last_layer = return_last_layer
        self.relu_act = nn.LeakyReLU(slope)
        self.model_type = model_type

        self.integrate_layers = nn.Sequential(
            self.get_fc_layer(seq_len * conv_in_channels, 2048, 2048, drop_p[0]),
            self.get_fc_layer(2048, mlp_mediate_neuron_num, mlp_mediate_neuron_num, drop_p[1]),
            self.get_fc_layer(mlp_mediate_neuron_num, mlp_feature_num, mlp_feature_num, drop_p[2])
        )
        if model_type == 'regression' or labels_num == 2:
            self.out_layer = nn.Linear(mlp_feature_num, labels_num, True)
        else:
            self.out_layer = nn.Linear(mlp_feature_num, labels_num * 2, True)
        self.mlp_out_neuron_num = labels_num

        # set device
        self.device0 = get_device(device_ids)
        self.to_device()

    def get_fc_layer(self, in_neuron, out_neuron, bn_d, drop_p):
        return nn.Sequential(
            nn.Dropout(drop_p), nn.Linear(in_neuron, out_neuron, False),
            nn.BatchNorm1d(bn_d), self.relu_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, channel, seq
        x3_n = self.integrate_layers(x.flatten(1, -1))  # x3: batch, mlp_feature_num
        if self.return_last_layer:
            if self.model_type == 'regression':
                x_output = self.out_layer(x3_n)  # x_output: batch, labels_num
            else:
                if self.mlp_out_neuron_num == 2:
                    x_output = softmax(self.out_layer(x3_n), dim=-1)  # x_output: batch, 2
                else:
                    x_output = softmax(
                        self.out_layer(x3_n).reshape(x3_n.shape[0], -1, 2), dim=-1
                    )  # x_output: batch, classes, 2
                    # x_output = x_output[:, :, 1] # x_output: batch, classes
            return x_output
        else:
            return x3_n

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            output = self.forward(x)
        self.train()
        return output

    def to_device(self):
        self.to(self.device0)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Union[None, List, float], gamma: float, num_classes: int, size_average: bool):
        super(FocalLoss, self).__init__()
        self.size_average = size_average

        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param preds: [batch, classes]
        :param labels: [batch, classes]
        :return: tensor
        """
        if label_mask is None:
            label_mask = torch.ones(preds.shape[0]).to(preds.device)
        label_mask = label_mask.to(preds.device)
        preds = preds[label_mask == 1]
        labels = labels[label_mask == 1]
        preds_softmax = preds[labels == 1]
        preds_logsoft = torch.log(preds_softmax.clamp(min=1e-10))
        alpha = self.alpha.to(preds.device)
        alpha = torch.tile(alpha, (preds.shape[0], 1))[labels == 1]
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
