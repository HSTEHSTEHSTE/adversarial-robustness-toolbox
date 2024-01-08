# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the task specific estimator for Icefall, an end-to-end speech recognition toolkit based on
k2-fsa.

| Repository link: https://github.com/k2-fsa/icefall/tree/master
"""
import ast
from argparse import Namespace
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from art import config
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin, PytorchSpeechRecognizerMixin
from art.utils import get_file

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    # import icefall - what's the role of type checking here?

    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)



class PyTorchIcefall(PytorchSpeechRecognizerMixin, SpeechRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speech recognizer using the end-to-end speech recognizer in
    Icefall.

    | Repository link: https://github.com/k2-fsa/icefall/tree/master
    """

    from icefall.utils import AttributeDict
    from pathlib import Path
    import k2
    estimator_params = PyTorchEstimator.estimator_params + ["icefall_config_filepath"]


    def __init__(
        self,
        icefall_config_filepath: Optional[str] = None,
        model: Optional[str] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
        verbose: bool = True,
    ):
        """
        Initialization of an instance PyTorchIcefall

        :param icefall_config_filepath: The path of the Icefall config file (yaml)
        :param model: The choice of pretrained model if a pretrained model is required.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch
        import yaml
        from transducer.decode import get_id2word

        self.icefall_config_filepath = icefall_config_filepath

        # Super initialization
        super().__init__(
            model=None,
            clip_values=clip_values,
            channels_first=None,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.verbose = verbose

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):  # pragma: no cover
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")
            if not np.all(self.clip_values[1] == 1):  # pragma: no cover
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")

        # Check postprocessing defences
        if self.postprocessing_defences is not None:  # pragma: no cover
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # Set cpu/gpu device
        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda", 0)

        # construct icefall args
        params = self.get_params()
        

        # load checkpoint# load_model_ensemble
        self.transducer_model = self.get_transducer_model(params)
        self.word2ids = self.get_word2id(params)
        self.get_id2word = get_id2word(params)


        if params.avg == 1:
            from icefall.checkpoint import load_checkpoint
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", self.transducer_model)
        else:
            from icefall.checkpoint import average_checkpoints
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if start >= 0:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            self.transducer_model.load_state_dict(average_checkpoints(filenames))


        
        self.transducer_model.to(self.device)


    def get_params(self) -> AttributeDict:
        """Return a dict containing training parameters.

        All training related parameters that are not passed from the commandline
        is saved in the variable `params`.

        Commandline options are merged into `params` after they are parsed, so
        you can also access them via `params`.

        Explanation of options saved in `params`:

            - lr: It specifies the initial learning rate

            - feature_dim: The model input dim. It has to match the one used
                        in computing features.

            - weight_decay:  The weight_decay for the optimizer.

            - subsampling_factor:  The subsampling factor for the model.

            - start_epoch:  If it is not zero, load checkpoint `start_epoch-1`
                            and continue training from that checkpoint.

            - best_train_loss: Best training loss so far. It is used to select
                            the model that has the lowest training loss. It is
                            updated during the training.

            - best_valid_loss: Best validation loss so far. It is used to select
                            the model that has the lowest validation loss. It is
                            updated during the training.

            - best_train_epoch: It is the epoch that has the best training loss.

            - best_valid_epoch: It is the epoch that has the best validation loss.

            - batch_idx_train: Used to writing statistics to tensorboard. It
                            contains number of batches trained so far across
                            epochs.

            - log_interval:  Print training loss if batch_idx % log_interval` is 0

            - valid_interval:  Run validation if batch_idx % valid_interval` is 0

            - reset_interval: Reset statistics if batch_idx % reset_interval is 0


        """
        from icefall.utils import AttributeDict
        from pathlib import Path
        params = AttributeDict(
            {
                "lr": 1e-3,
                "feature_dim": 23,
                "weight_decay": 1e-6,
                "start_epoch": 0,
                "best_train_loss": float("inf"),
                "best_valid_loss": float("inf"),
                "best_train_epoch": -1,
                "best_valid_epoch": -1,
                "batch_idx_train": 0,
                "log_interval": 100,
                "reset_interval": 20,
                "valid_interval": 300,
                "exp_dir": Path("transducer/exp_lr1e-4"),
                "lang_dir": Path("data/lm/frames"),
                # encoder/decoder params
                "vocab_size": 3,  # blank, yes, no
                "blank_id": 0,
                "embedding_dim": 32,
                "hidden_dim": 16,
                "num_decoder_layers": 4,
                "epoch": 1,
                "avg": 1
            }
        )

        vocab_size = 1
        with open(Path(params.lang_dir) / 'lexicon_disambig.txt') as lexicon_file:
            for line in lexicon_file:
                if len(line.strip()) > 0:# and '<UNK>' not in line and '<s>' not in line and '</s>' not in line:
                    vocab_size += 1
        params.vocab_size = vocab_size

        return params

    def predict(self, x: np.ndarray, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :return: Transcription as a numpy array of characters. A possible example of a transcription return
                 is `np.array(['SIXTY ONE', 'HELLO'])`.
        """
        from transducer.beam_search import greedy_search

        assert batch_size == 1

        x_in = np.empty(len(x), dtype=object)
        x_in[:] = list(x)
        assert len(x) == 1

        # Put the model in the eval mode
        self.transducer_model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_in, y=None, fit=False)

        # Run prediction with batch processing
        decoded_output = []
        # result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for sample_index in range(num_batch):
            wav = x_preprocessed[sample_index] # np.array, len = wav len
            shape = wav.shape

            # extract features
            x = self.transform_model_input(x=torch.tensor(wav))

            print(shape)
            encoder_out, encoder_out_lens = self.transducer_model.encoder(x=x, x_lens=shape)
            hyp = greedy_search(model=self.transducermodel, encoder_out=encoder_out, id2word=self.get_id2word)
            decoded_output.append(hyp)
            
        return np.concatenate(decoded_output)

    def get_transducer_model(self, params: AttributeDict):
        from transducer.decoder import Decoder
        from transducer.conformer import Conformer
        from transducer.joiner import Joiner
        from transducer.model import Transducer
        encoder = Conformer(
            num_features=params.feature_dim,
            output_dim=params.hidden_dim,
        )
        decoder = Decoder(
            vocab_size=params.vocab_size,
            embedding_dim=params.embedding_dim,
            blank_id=params.blank_id,
            num_layers=params.num_decoder_layers,
            hidden_dim=params.hidden_dim,
            embedding_dropout=0.4,
            rnn_dropout=0.4,
        )
        joiner = Joiner(input_dim=params.hidden_dim, output_dim=params.vocab_size)
        transducer = Transducer(encoder=encoder, decoder=decoder, joiner=joiner)

        return transducer

    def get_word2id(self, params):
        from pathlib import Path
        word2id = {}

        # 0 is blank
        id = 1
        with open(Path(params.lang_dir) / 'lexicon_disambig.txt') as lexicon_file:
            for line in lexicon_file:
                if len(line.strip()) > 0:
                    word2id[line.split()[0]] = id
                    id += 1

        return word2id 

    def loss_gradient(self, x, y: np.ndarray, **kwargs) -> np.ndarray:
        import k2

        x = torch.autograd.Variable(x, requires_grad=True)
        features, _, _ = self.transform_model_input(x=x, compute_gradient=True)
        x_lens = torch.tensor([features.shape[1]]).to(torch.int32).to(self.device)
        y = k2.RaggedTensor(y)
        loss = self.transducer_model(x=features, x_lens=x_lens, y=y)
        loss.backward()

        # Get results
        results = x.grad
        results = self._apply_preprocessing_gradient(x, results)
        return results

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the estimator on the training set `(x, y)`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        raise NotImplementedError

    def transform_model_input(
            self,
            x,
            y=None,
            compute_gradient=False
    ):
        """
        Transform the user input space into the model input space.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param compute_gradient: Indicate whether to compute gradients for the input `x`.
        :param tensor_input: Indicate whether input is tensor.
        :param real_lengths: Real lengths of original sequences.
        :return: A tupe of a sorted input feature tensor, a supervision tensor,  and a list representing the original order of the batch
        """
        import torch  # lgtm [py/repeated-import]
        import torchaudio

        from dataclasses import dataclass, asdict
        @dataclass
        class FbankConfig:
            # Spectogram-related part
            dither: float = 0.0
            window_type: str = "povey"
            # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
            frame_length: float = 0.025
            frame_shift: float = 0.01
            remove_dc_offset: bool = True
            round_to_power_of_two: bool = True
            energy_floor: float = 1e-10
            min_duration: float = 0.0
            preemphasis_coefficient: float = 0.97
            raw_energy: bool = True
            
            # Fbank-related part
            low_freq: float = 20.0
            high_freq: float = -400.0
            num_mel_bins: int = 40
            use_energy: bool = False
            vtln_low: float = 100.0
            vtln_high: float = -500.0
            vtln_warp: float = 1.0

        params = asdict(FbankConfig())
        params.update({
            "sample_frequency": 16000,
            "snip_edges": False,
            "num_mel_bins": 23
        })
        params['frame_shift'] *= 1000.0
        params['frame_length'] *= 1000.0
        

        feature_list = []
        num_frames = []
        supervisions = {}

        for i in range(len(x)):
            isnan = torch.isnan(x[i])
            nisnan=torch.sum(isnan).item()
            if nisnan > 0:
                logging.info('input isnan={}/{} {}'.format(nisnan, x[i].shape, x[i][isnan], torch.max(torch.abs(x[i]))))


            xx = x[i]
            xx = xx.to(self._device)
            feat_i = torchaudio.compliance.kaldi.fbank(xx.unsqueeze(0), **params) # [T, C]
            feat_i = feat_i.transpose(0, 1) #[C, T]
            feature_list.append(feat_i)
            num_frames.append(feat_i.shape[1])
        
        indices = sorted(range(len(feature_list)),
                         key=lambda i: feature_list[i].shape[1], reverse=True)
        indices = torch.LongTensor(indices)
        num_frames = torch.IntTensor([num_frames[idx] for idx in indices])
        start_frames = torch.zeros(len(x), dtype=torch.int)

        supervisions['sequence_idx'] = indices.int()
        supervisions['start_frame'] = start_frames
        supervisions['num_frames'] = num_frames
        if y is not None:
            supervisions['text'] = [y[idx] for idx in indices]

        feature_sorted = [feature_list[index] for index in indices]
        
        feature = torch.zeros(len(feature_sorted), feature_sorted[0].size(0), feature_sorted[0].size(1), device=self._device)

        for i in range(len(x)):
            feature[i, :, :feature_sorted[i].size(1)] = feature_sorted[i]

        return feature.transpose(1, 2), supervisions, indices


    def to_training_mode(self) -> None:
        """
        Put the estimator in the training mode.
        """
        self.transducer_model.train()

    @property
    def sample_rate(self) -> int:
        """
        Get the sampling rate.

        :return: The audio sampling rate.
        """
        return self._sampling_rate

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def model(self):
        """
        Get current model.

        :return: Current model.
        """
        return self._model

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def compute_loss_and_decoded_output(
        self, masked_adv_input: "torch.Tensor", original_output: np.ndarray, **kwargs
    ) -> Tuple["torch.Tensor", np.ndarray]:
        """
        Compute loss function and decoded output.

        :param masked_adv_input: The perturbed inputs.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: The loss and the decoded output.
        """
        from transducer.beam_search import greedy_search
        import k2
        
        assert len(original_output[0]) == 1
        num_batch = len(original_output[0])
        decoded_output = []

        for sample_index in range(num_batch):
            features, _, _ = self.transform_model_input(x=masked_adv_input[sample_index])
            x_lens = torch.tensor([features.shape[1]]).to(torch.int32).to(self.device)
            y = k2.RaggedTensor(original_output[sample_index])
            loss = self.transducer_model(x=features, x_lens=x_lens, y=y)

            encoder_out, encoder_out_lens = self.transducer_model.encoder(x=features, x_lens=masked_adv_input[sample_index].shape)
            hyp = greedy_search(model=self.transducermodel, encoder_out=encoder_out, id2word=self.get_id2word)
            decoded_output.append(hyp)
            
        return np.concatenate(decoded_output)