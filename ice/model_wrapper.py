"""
@Author: Zhang Ruihan
@Date: 2019-10-28 01:01:52
@LastEditors: Thao Le
@Description: file content
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import params


class PytorchModelWrapper:
    def __init__(
        self,
        model,
        layer_dict={},
        predict_target=None,
        input_channel_first=True,  # True if input image is channel first
        model_channel_first=True,  # True if model use channel first
        numpy_out=True,
        input_size=[3, 224, 224],  # model's input size
        batch_size=128,
    ):
        self.model = model
        self.layer_dict = layer_dict
        self.layer_dict.update(dict(model.named_children()))
        self.predict_target = predict_target
        self.input_channel = "f" if input_channel_first else "l"
        self.model_channel = "f" if model_channel_first else "l"
        self.numpy_out = numpy_out
        self.input_size = list(input_size)
        self.batch_size = batch_size
        self.non_negative = False

    def _to_tensor(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = torch.clone(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x

    # transform from channel first to channel last
    def _switch_channel_f_to_l(self, x):
        if x.ndim == 3:
            x = x.permute(1, 2, 0)
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1)

        return x

    # transform from channel last to channel first
    def _switch_channel_l_to_f(self, x):
        if x.ndim == 3:
            x = x.permute(2, 0, 1)
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        return x

    def _switch_channel(self, x, layer_in="input", to_model=True):
        c_from = None
        c_to = None
        if to_model:
            c_from = self.input_channel if layer_in == "input" else "l"
            c_to = self.model_channel
        else:
            c_from = self.model_channel
            c_to = "l"

        if c_from == "f" and c_to == "l":
            x = self._switch_channel_f_to_l(x)
        if c_from == "l" and c_to == "f":
            x = self._switch_channel_l_to_f(x)
        return x

    def _input_to_output(self, x, layer_in="input", layer_out="output"):
        # tensor cpu in cpu out

        x = x.type(torch.FloatTensor)
        data_in = x.clone()
        data_in = data_in.to(params.DEVICE)
        data_out = []

        handles = []

        def hook_in(m, i, o):
            return data_in

        def hook_out(m, i, o):
            data_out.append(o)

        if layer_in == "input":
            nx = x
        else:
            handles.append(self.layer_dict[layer_in].register_forward_hook(hook_in))
            nx = torch.zeros([x.size()[0]] + self.input_size)

        if not layer_out == "output":
            handles.append(self.layer_dict[layer_out].register_forward_hook(hook_out))

        nx = nx.to(params.DEVICE)
        with torch.no_grad():
            ny = self.model(nx)

        if layer_out == "output":
            data_out = ny
        else:
            data_out = data_out[0]

        data_out = data_out.cpu()

        for handle in handles:
            handle.remove()

        if self.non_negative:
            data_out = torch.relu(data_out)

        return data_out

    def _batch_fn(self, x, layer_in="input", layer_out="output"):
        # numpy in numpy out

        if type(x) == torch.Tensor or type(x) == np.ndarray:
            x = self._to_tensor(x)

            dataset = TensorDataset(x)
            x = DataLoader(dataset, batch_size=self.batch_size)

        out = []

        for nx in x:
            nx = nx[0]
            nx = self._switch_channel(nx, layer_in=layer_in, to_model=True)
            out.append(self._input_to_output(nx, layer_in, layer_out))

        res = torch.cat(out, 0)

        res = self._switch_channel(res, layer_in=layer_in, to_model=False)
        if self.numpy_out:
            res = res.detach().numpy()

        return res

    def set_predict_target(self, predict_target):
        self.predict_target = predict_target

    def get_feature(self, x, layer_name):
        if layer_name not in self.layer_dict:
            print("Target layer not exists")
            return None

        out = self._batch_fn(x, layer_out=layer_name)

        return out

    def feature_predict(self, feature, layer_name=None):
        if layer_name not in self.layer_dict:
            print("Target layer not exists")
            return None

        out = self._batch_fn(feature, layer_in=layer_name)
        if self.predict_target is not None:
            out = out[:, self.predict_target]
        return out

    def predict(self, x):
        out = self._batch_fn(x)
        if self.predict_target is not None:
            out = out[:, self.predict_target]

        return out
