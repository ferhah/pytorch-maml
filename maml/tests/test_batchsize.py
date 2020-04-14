import pytest
from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning
from maml.utils import tensors_to_device
from torchmeta.utils.data import BatchMetaDataLoader
import torch

def test_batchsize():
    num_steps = 1
    num_workers = 0
    dataset = 'miniimagenet'
    folder = 'data/miniimagenet'
    num_ways = 4
    num_shots = 4
    num_shots_test = 4
    hidden_size = 64
    batch_size = 5
    first_order = False
    step_size = 0.1
    benchmark = get_benchmark_by_name(dataset,
                                      folder,
                                      num_ways,
                                      num_shots,
                                      num_shots_test,
                                      hidden_size=hidden_size)
    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            first_order=first_order,
                                            num_adaptation_steps=num_steps,
                                            step_size=step_size,
                                            loss_function=benchmark.loss_function,
                                            device='cpu')
    for batch in meta_test_dataloader:
        batch = tensors_to_device(batch, device='cpu')
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(zip(*batch['train'], *batch['test'])):
            params, _ = metalearner.adapt(train_inputs, train_targets,
                                                    is_classification_task=True,
                                                    num_adaptation_steps=metalearner.num_adaptation_steps,
                                                    step_size=metalearner.step_size, first_order=metalearner.first_order)
            test_logits_1 = metalearner.model(test_inputs, params=params)
            for idx in range(test_inputs.shape[0]):
                test_logits_2 = metalearner.model(test_inputs[idx:idx + 1, ...], params=params)
                assert torch.allclose(test_logits_1[idx:idx + 1, ...], test_logits_2, atol=1e-04)
        break
    return