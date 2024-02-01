# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements Clean Label Backdoor Attacks to poison data used in ML models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING, Union, Callable, List

import numpy as np
import torch

from art.attacks.attack import PoisoningAttackBlackBox
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor

if TYPE_CHECKING:
    from art.utils import ESTIMATOR_TYPE


logger = logging.getLogger(__name__)


class PoisoningAttackCleanLabelEstimatorBackdoor(PoisoningAttackBlackBox):
    """
    Implementation of Clean-Label Backdoor Attack introduced in Turner et al., 2018.

    Applies a number of backdoor perturbation functions and does not change labels.

    | Paper link: https://people.csail.mit.edu/madry/lab/cleanlabel.pdf
    """

    attack_params = PoisoningAttackBlackBox.attack_params + [
        "backdoor",
        "proxy_classifier",
        "target",
        "pp_poison",
        "norm",
        "eps",
        "eps_step",
        "max_iter",
        "num_random_init",
        "source"
    ]
    _estimator_requirements = ()

    def __init__(
        self,
        backdoor: PoisoningAttackBackdoor,
        proxy_estimator: "ESTIMATOR_TYPE",
        target_function: Callable,
        target_transformation_function: Callable = None,
        pp_poison: float = 0.33,
        norm: Union[int, float, str] = np.inf,
        eps: Union[float, Callable] = 0.3,
        eps_step: [float, Callable] = 0.1,
        max_iter: int = 100,
        num_random_init: int = 0,
    ) -> None:
        """
        Creates a new Clean Label Backdoor poisoning attack

        :param backdoor: the backdoor chosen for this attack
        :param proxy_estimator: the estimator for this attack
        :param target_function: returns True if an input is a poisoning target, False otherwise
        :param target_transformation_function: returns target label for adversarial perturbation. None if adversarial perturbation is untargeted
        :param pp_poison: The percentage of the data to poison. Note: Only data within the target label is poisoned
        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce. Can be fixed float value, or dynamic based on
                    input x.
        :param eps_step: Attack step size (input variation) at each iteration. Can be fixed float value, or dynamic based
                         on input x
        :param max_iter: The maximum number of iterations.
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        """
        super().__init__()
        self.backdoor = backdoor
        self.proxy_estimator = proxy_estimator
        self.target_function = target_function
        self.target_transformation_function = target_transformation_function
        self.pp_poison = pp_poison
        self.eps = eps
        self.eps_step = eps_step
        if self.target_transformation_function is None: # default: untargeted attack
            self.targeted = False
            self.attack = ProjectedGradientDescent(
                proxy_estimator,
                norm=norm,
                eps=0.33,
                eps_step=0.1,
                max_iter=max_iter,
                targeted=self.targeted,
                num_random_init=num_random_init,
                batch_size = 1,
                compute_success = False
            )
        else:
            self.targeted = True
            self.attack = ProjectedGradientDescent(
                proxy_estimator,
                targeted=self.targeted,
                norm=norm,
                eps=0.33,
                eps_step=0.1,
                max_iter=max_iter,
                num_random_init=num_random_init,
                batch_size = 1,
                compute_success = False
            )
        self._check_params()

    def poison(  # pylint: disable=W0221
        self, 
        x: Union[np.ndarray, List],
        y: Union[np.ndarray, List] = None,
        y_adv: Union[np.ndarray, List] = None,
        broadcast: bool = True, 
        **kwargs
    ) -> Tuple[List, List]:
        """
        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.

        :param x: An array with the points that initialize attack points.
        :param y: The target labels for the attack.
        :param y_adv: The target labels for the pgd attack.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        if isinstance(x, np.ndarray):
            x = x.tolist()
        data = x.copy()

        if y is None:
            estimated_labels = []
            for x_instance in x:
                estimated_labels.append(self.proxy_estimator.predict(np.expand_dims(x_instance, 0)))
        else:
            if isinstance(y, np.ndarray):
                y = y.tolist()
            estimated_labels = y.copy()

        # Selected target indices to poison
        target_indices = []
        for x_index, x_instance in enumerate(x):
            if self.target_function(estimated_labels[x_index]):
                target_indices.append(x_index)
        target_indices = np.array(target_indices)
        num_poison = int(self.pp_poison * len(target_indices))
        selected_indices = np.random.choice(target_indices, num_poison)

        # Run untargeted PGD on selected points, making it hard to classify correctly
        perturbed_inputs = []
        if self.targeted: # targeted attack
            if y_adv is None:
                assert self.target_transformation_function is not None
                for selected_index in selected_indices:
                    eps = self.eps(data[selected_index])
                    eps_step = self.eps_step(data[selected_index])
                    self.attack.set_params(eps=eps, eps_step=eps_step)
                    pgd_target = self.target_transformation_function(estimated_labels[selected_index])
                    perturbed_input = self.attack.generate(data[selected_index], pgd_target)
                    perturbed_inputs.append(perturbed_input)
            else:
                for selected_index in selected_indices:
                    eps = self.eps(data[selected_index])
                    eps_step = self.eps_step(data[selected_index])
                    self.attack.set_params(eps=eps, eps_step=eps_step)
                    perturbed_input = self.attack.generate(data[selected_index], y_adv[selected_index])
                    perturbed_inputs.append(perturbed_input)
        else: # untargeted attack
            for selected_index in selected_indices:
                eps = self.eps(data[selected_index])
                eps_step = self.eps_step(data[selected_index])
                self.attack.set_params(eps=eps, eps_step=eps_step)
                perturbed_input = self.attack.generate(data[selected_index])
                perturbed_inputs.append(perturbed_input)

        # Add backdoor and poison with the same label
        for perturbed_index, selected_index in enumerate(selected_indices):
            poisoned_input, _ = self.backdoor.poison(perturbed_inputs[perturbed_index], estimated_labels[perturbed_index], broadcast=broadcast)
            data[selected_index] = poisoned_input

        return data, estimated_labels

    def _check_params(self) -> None:
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError("Backdoor must be of type PoisoningAttackBackdoor")
        if not isinstance(self.attack, ProjectedGradientDescent):
            raise ValueError("There was an issue creating the PGD attack")
        if not 0 <= self.pp_poison <= 1:
            raise ValueError("pp_poison must be between 0 and 1")
