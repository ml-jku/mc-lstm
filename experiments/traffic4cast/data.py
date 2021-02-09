from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class Traffic4Cast20(Dataset):
    """ Data for the 2020 traffic4cast challenge. """

    URLS = {
        "https://zurich-ml-datasets.s3.amazonaws.com/traffic4cast/2020/BERLIN.tar?AWSAccessKeyId=AKIAYKRMCYB5YMM4ZCES&Expires=1609923309&Signature=hqE1F%2FkqZmozMMLEEGJUjnYIppo%3D&mkt_tok=eyJpIjoiWkRSak1tRmpNalZtWlRWbSIsInQiOiJubGQzRzBzanhWVkRFQk1mNXJMRytwY0NHaklGelZPY3lVS3RMN1UyQkk2ZUJZSUxLQmtvTGpXVlFLNFhlUFhUN2g3UzZFWFpVUWlVZFJZZXFmOE01WXBXMFNJRjdlU3VBTG9vZDhydTNiRHE0V0RMVGRsdkRic1pSOUJUMWErUCJ9":
            {
                'filename': "BERLIN.tar",
                'md5': "e5ff2ea5bfad2c7098aa1e58e21927a8"
            },
        "https://zurich-ml-datasets.s3.amazonaws.com/traffic4cast/2020/MOSCOW.tar?AWSAccessKeyId=AKIAYKRMCYB5YMM4ZCES&Expires=1609923367&Signature=MKSDRXlbu0mgpOTsh8Lg6WbOK%2FI%3D&mkt_tok=eyJpIjoiTVROaE1ERTFOR0ZsTXpVeCIsInQiOiJyaUIzTUhVSUNucE9cL0J1U1BaNVZTUWE3WVJud3VFbDRFbW40ZHRJdzJldVo2VWoyZFdJSWJEWjQ1TjZla2NcL3NWVVJ6bkNTREdoN0R6cmVQRTlrTGhwZWorMjJNUTdkYVVFb3dRelVETENsMUppR3FDeGFvcmZXeDJjVkRyUmJtIn0":
            {
                'filename': "MOSCOW.tar",
                'md5': "8101753853af80c183f3cc10f84d42f4"
            },
        "https://zurich-ml-datasets.s3.amazonaws.com/traffic4cast/2020/ISTANBUL.tar?AWSAccessKeyId=AKIAYKRMCYB5YMM4ZCES&Expires=1609923358&Signature=BSbSFV0%2B%2F5VeU9d0uFFeJiGWuFg%3D&mkt_tok=eyJpIjoiTVROaE1ERTFOR0ZsTXpVeCIsInQiOiJyaUIzTUhVSUNucE9cL0J1U1BaNVZTUWE3WVJud3VFbDRFbW40ZHRJdzJldVo2VWoyZFdJSWJEWjQ1TjZla2NcL3NWVVJ6bkNTREdoN0R6cmVQRTlrTGhwZWorMjJNUTdkYVVFb3dRelVETENsMUppR3FDeGFvcmZXeDJjVkRyUmJtIn0":
            {
                'filename': "ISTANBUL.tar",
                'md5': "229730cf5e95d31d1e8494f2a57e50e9"
            }
    }

    def __init__(self,
                 root: str,
                 train: bool = True,
                 city: str = None,
                 single_sample: bool = False,
                 normalised: bool = False,
                 time_diff: int = 0,
                 masking: str = None,
                 sparse: bool = False,
                 download: bool = False,
                 seed: int = None):

        self.root = Path(root).expanduser().resolve()
        self.normalised = normalised
        self.time_diff = time_diff
        self.mask = None
        self.sparse = sparse

        if download:
            self.download()

        city = "" if city is None else city.upper()
        file_id = city if city in {"BERLIN", "ISTANBUL", "MOSCOW"} else "*"
        mode = "training" if train else "validation"
        data = []
        for file in self.path.glob("_".join([file_id, mode]) + ".npz"):
            data.append(np.load(file.with_suffix(".npz")))

        self.inputs = np.concatenate([arr['ins'] for arr in data], axis=0)
        self.outputs = np.concatenate([arr['outs'] for arr in data], axis=0)

        if single_sample:
            self.inputs = self.inputs.reshape(1, -1, self.inputs.shape[-1])
            self.outputs = self.outputs.reshape(1, -1, self.outputs.shape[-1])

        if masking == 'zero':
            self.mask = np.zeros_like(self.inputs[0])
        elif masking == 'avg' or sparse:
            if single_sample:
                raise NotImplementedError("no meaningful averaging for single seq (yet)")
            self.mask = np.mean(self.inputs, axis=0)

        if sparse:
            rng = np.random.RandomState(seed)
            self._indices = rng.randint(self.inputs.shape[1], size=self.inputs.shape[:1])

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.inputs[index].astype('float32'))
        outputs = torch.from_numpy(self.outputs[index].astype('float32'))
        aux = torch.linspace(0., 1., 288).repeat(len(inputs) // 288).view(-1, 1)

        if self.sparse:
            xi = inputs[self.idx]
            inputs[:] = torch.from_numpy(self.mask)
            inputs[self.idx] = xi
        elif self.mask is not None:
            mask_val = torch.from_numpy(self.mask[len(inputs) - self.time_diff:])
            inputs[len(inputs) - self.time_diff:] = mask_val
        elif self.time_diff:
            inputs = inputs[:len(inputs) - self.time_diff]
            aux = aux[:len(inputs)]
            outputs = outputs[self.time_diff:]

        if self.normalised:
            mu, sigma = inputs.mean(), inputs.std()
            inputs = (inputs - mu) / sigma
            outputs = (outputs - mu) / sigma

        return inputs, aux, outputs

    def __len__(self):
        return len(self.inputs)

    @property
    def idx(self):
        if self.sparse:
            # 6AM, 12AM, 6PM +- 5 min
            return [71, 72, 73, 143, 144, 145, 215, 216, 217]
        else:
            # midnight
            return -1

    @property
    def path(self):
        return self.root / "traffic4cast20"

    @staticmethod
    def _process_dir(path: Path):
        inputs, outputs = [], []
        for file in sorted(path.glob("*.h5")):
            with h5py.File(file, 'r') as f:
                data = np.asarray(f['array'])

            volumes = data[..., :8:2]  # last column: NE, NW, SE, SW
            north_south = np.sum(volumes[..., [0, -1], 1:-1, :], axis=-2)
            west_east = np.sum(volumes[..., 1:-1, [0, -1], :], axis=-3)
            corners = volumes[..., [0, -1, 0, -1], [0, 0, -1, -1], :]
            nw, sw, ne, se = np.moveaxis(corners, -2, 0)

            incoming = [
                np.sum(west_east[..., 0, 0::2], axis=-1) + (sw[..., 0] + nw[..., 2]) / 2,  # W
                np.sum(west_east[..., 1, 1::2], axis=-1) + (se[..., 1] + ne[..., 3]) / 2,  # E
                np.sum(north_south[..., 0, 2:], axis=-1) + (nw[..., 2] + ne[..., 3]) / 2,  # N
                np.sum(north_south[..., 1, :2], axis=-1) + (sw[..., 0] + se[..., 1]) / 2  # S
            ]

            outgoing = [
                np.sum(west_east[..., 0, 1::2], axis=-1) + sw[..., 1] + nw[..., 3] + (sw[..., 3] + nw[..., 1]) / 2,  # W
                np.sum(west_east[..., 1, 0::2], axis=-1) + se[..., 0] + ne[..., 2] + (se[..., 2] + ne[..., 0]) / 2,  # E
                np.sum(north_south[..., 0, :2], axis=-1) + nw[..., 0] + ne[..., 1] + (nw[..., 1] + ne[..., 0]) / 2,  # N
                np.sum(north_south[..., 1, 2:], axis=-1) + sw[..., 2] + se[..., 3] + (sw[..., 3] + se[..., 2]) / 2  # S
            ]

            xi = np.stack(incoming, axis=-1)
            yi = np.stack(outgoing, axis=-1)
            inputs.append(xi)
            outputs.append(yi)

        if data.ndim == 4:
            return np.stack(inputs), np.stack(outputs)
        else:
            return np.concatenate(inputs), np.concatenate(outputs)

    def download(self):
        raw_dir = self.path / "raw"
        for url, kwargs in self.URLS.items():
            download_and_extract_archive(url, str(self.path), extract_root=str(raw_dir), **kwargs)

            file_name = kwargs["filename"].rsplit(".", 1)[0]
            for subdir in ["training", "validation"]:
                file_path = self.path / "_".join([file_name, subdir])
                x, y = self._process_dir(raw_dir / file_name / subdir)
                np.savez(str(file_path.with_suffix(".npz")), ins=x, outs=y)
