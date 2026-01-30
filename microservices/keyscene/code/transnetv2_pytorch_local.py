from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import tensorflow as tf


class TransNetV2(nn.Module):
    def __init__(
        self,
        F: int = 16,
        L: int = 3,
        S: int = 2,
        D: int = 1024,
        use_many_hot_targets: bool = True,
        use_frame_similarity: bool = True,
        use_color_histograms: bool = True,
        use_mean_pooling: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.use_mean_pooling = use_mean_pooling
        self.use_frame_similarity = use_frame_similarity
        self.use_color_histograms = use_color_histograms

        self.SDDCNN = nn.ModuleList(
            [
                StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0),
                *[
                    StackedDDCNNV2(
                        in_filters=(F * 2 ** (i - 1)) * 4,
                        n_blocks=S,
                        filters=F * 2 ** i,
                        stochastic_depth_drop_prob=0.0,
                    )
                    for i in range(1, L)
                ],
            ]
        )

        self.frame_sim_layer = (
            FrameSimilarity(
                sum([(F * 2 ** i) * 4 for i in range(L)]),
                lookup_window=101,
                output_dim=128,
                similarity_dim=128,
                use_bias=True,
            )
            if use_frame_similarity
            else None
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot


class StackedDDCNNV2(nn.Module):
    def __init__(self, in_filters, n_blocks, filters, shortcut=True, pool_type="avg", stochastic_depth_drop_prob=0.0):
        super().__init__()
        assert pool_type in {"max", "avg"}
        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None):
        super().__init__()
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation_rate, separable=True, use_bias=True):
        super().__init__()
        assert not separable is False or True

        if separable:
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, similarity_dim=128, lookup_window=101, output_dim=128, use_bias=False):
        super().__init__()
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)
        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))
        pad = (self.lookup_window - 1) // 2
        similarities_padded = functional.pad(similarities, [pad, pad])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window]
        )
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window]
        )
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]
        ) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super().__init__()
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames_in):
            r, g, b = frames_in[:, :, 0], frames_in[:, :, 1], frames_in[:, :, 2]
            r, g, b = r >> 5, g >> 5, b >> 5
            return (r << 6) + (g << 3) + b

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms = functional.normalize(histograms, p=2, dim=2)
        return histograms

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))
        pad = (self.lookup_window - 1) // 2
        similarities_padded = functional.pad(similarities, [pad, pad])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window]
        )
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window]
        )
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]
        ) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities


# ---- TF -> Torch conversion ----

def _remap_name(x: str) -> str:
    x = x.replace("TransNet/", "")
    l = []
    for a in x.split("/"):
        if a.startswith("SDDCNN") or a.startswith("DDCNN"):
            a = a.split("_")
            a = a[0] + "." + str(int(a[1]) - 1)
        elif a == "conv_spatial":
            a = "layers.0"
        elif a == "conv_temporal":
            a = "layers.1"
        elif a == "kernel:0" or a == "gamma:0":
            a = "weight"
        elif a == "bias:0" or a == "beta:0":
            a = "bias"
        elif a == "dense":
            a = "fc1"
        elif a == "dense_1":
            a = "cls_layer1"
        elif a == "dense_2":
            a = "cls_layer2"
        elif a == "dense_3":
            a = "frame_sim_layer.projection"
        elif a == "dense_4":
            a = "frame_sim_layer.fc"
        elif a == "dense_5":
            a = "color_hist_layer.fc"
        elif a in {"FrameSimilarity", "ColorHistograms"}:
            a = ""
        elif a == "moving_mean:0":
            a = "running_mean"
        elif a == "moving_variance:0":
            a = "running_var"
        l.append(a)
    x = ".".join([a for a in l if a != ""])
    return x


def _remap_tensor(x):
    x = x.numpy()
    if len(x.shape) == 5:
        x = np.transpose(x, [0, 1, 2, 4, 3])
        x = np.transpose(x, [3, 4, 0, 1, 2])
    elif len(x.shape) == 2:
        x = np.transpose(x)
    return torch.from_numpy(x).clone()


def _check_and_fix_dicts(tf_dict: Dict[str, torch.Tensor], torch_dict: Dict[str, Tuple[int, ...]]) -> bool:
    error = False
    for k in torch_dict.keys():
        if k not in tf_dict:
            if k.endswith("num_batches_tracked"):
                tf_dict[k] = torch.tensor(1.0, dtype=torch.float32)
            else:
                error = True
    for k in tf_dict.keys():
        if k not in torch_dict:
            error = True
        elif tuple(tf_dict[k].shape) != torch_dict[k]:
            error = True
    return not error


def convert_tf_weights(tf_weights_dir: str) -> TransNetV2:
    tf_model = tf.saved_model.load(tf_weights_dir)
    tf_dict = {_remap_name(v.name): _remap_tensor(v) for v in tf_model.variables}

    torch_model = TransNetV2()
    torch_dict = {k: tuple(v.shape) for k, v in list(torch_model.named_parameters()) + list(torch_model.named_buffers())}

    if not _check_and_fix_dicts(tf_dict, torch_dict):
        raise RuntimeError("TransNetV2 weight conversion failed: shape mismatch")

    torch_model.load_state_dict(tf_dict)
    return torch_model


def load_or_convert_weights(tf_weights_dir: str, torch_weights_path: str) -> TransNetV2:
    torch_path = Path(torch_weights_path)
    if torch_path.exists():
        model = TransNetV2()
        state = torch.load(str(torch_path), map_location="cpu")
        model.load_state_dict(state)
        return model

    model = convert_tf_weights(tf_weights_dir)
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(torch_path))
    return model


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t_prev = 0
    start = 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)
