#!/usr/bin/env python
'''
Test script for EDOT_disc_torch.py

This script tests the PyTorch implementation of EDOT discretization functions.
'''

import torch
import numpy as np
from src import EDOT_disc_torch as edt
from src import EDOT_discretization as ed


def test_euclidean_distance():
    # Create random sample and target points
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sample and target points
    sample_torch = torch.rand(5, 2)
    target_torch = torch.rand(3, 2)

    # Convert to numpy for comparison
    sample_numpy = sample_torch.cpu().numpy()
    target_numpy = target_torch.cpu().numpy()

    # Calculate distances
    dist_torch = edt.euclidean_distance(sample_torch, target_torch)
    dist_numpy = ed.euclidean_distance(sample_numpy, target_numpy)

    # Compare results
    print("Euclidean Distance Test:")
    print("PyTorch:")
    print(dist_torch)
    print("NumPy:")
    print(dist_numpy)
    print("Max Difference:",
          torch.max(torch.abs(torch.tensor(dist_numpy) - dist_torch)).item())
    print()


def test_gradient_EOT():
    # Create random sample and target points
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sample and target points
    sample_size, target_size, dim = 4, 3, 2
    sample_torch = torch.rand(sample_size, dim)
    target_torch = torch.rand(target_size, dim)
    sample_weight_torch = torch.ones(sample_size) / sample_size
    target_weight_torch = torch.ones(target_size) / target_size

    # Convert to numpy for comparison
    sample_numpy = sample_torch.numpy()
    target_numpy = target_torch.numpy()
    sample_weight_numpy = sample_weight_torch.numpy()
    target_weight_numpy = target_weight_torch.numpy()

    # Calculate gradient_EOT
    zeta = 0.1
    D_nu_obj_torch, D_y_obj_torch, cost_torch = edt.gradient_EOT(
        sample_torch,
        sample_weight_torch,
        target_torch,
        target_weight_torch,
        zeta=zeta)

    D_nu_obj_numpy, D_y_obj_numpy, cost_numpy = ed.gradient_EOT(
        sample_numpy,
        sample_weight_numpy,
        target_numpy,
        target_weight_numpy,
        zeta=zeta)

    # Compare results
    print("Gradient EOT Test:")
    print("PyTorch Cost:", cost_torch.item())
    print("NumPy Cost:", cost_numpy)
    print("Cost Difference:", abs(cost_torch.item() - cost_numpy))
    print()
    print("PyTorch D_nu_obj:")
    print(D_nu_obj_torch)
    print("NumPy D_nu_obj:")
    print(D_nu_obj_numpy)
    print(
        "Max D_nu_obj Difference:",
        torch.max(torch.abs(torch.tensor(D_nu_obj_numpy) -
                            D_nu_obj_torch)).item())
    print()
    print("PyTorch D_y_obj:")
    print(D_y_obj_torch)
    print("NumPy D_y_obj:")
    print(D_y_obj_numpy)
    print(
        "Max D_y_obj Difference:",
        torch.max(torch.abs(torch.tensor(D_y_obj_numpy) -
                            D_y_obj_torch)).item())
    print()


def test_cont_wasserstein():
    # Create random target points
    torch.manual_seed(42)
    np.random.seed(42)

    # Create target points
    target_size = 5
    target_torch = torch.rand(
        target_size, 1, device="cuda" if torch.cuda.is_available() else "cpu")
    target_weight_torch = torch.ones(target_size,
                                     device=target_torch.device) / target_size

    # Convert to numpy for comparison
    target_numpy = target_torch.cpu().numpy()
    target_weight_numpy = target_weight_torch.cpu().numpy()

    # Calculate cont_wasserstein
    N = 100
    zeta = 0.01
    W_inf_torch, W_list_torch = edt.cont_wasserstein(edt.uniform_sampler,
                                                     target_torch,
                                                     target_weight_torch,
                                                     N=N,
                                                     zeta=zeta)

    W_inf_numpy, W_list_numpy = ed.cont_wasserstein(ed.uniform_sampler,
                                                    target_numpy,
                                                    target_weight_numpy,
                                                    N=N,
                                                    zeta=zeta)

    # Compare results
    print("Continuous Wasserstein Test:")
    print("PyTorch W_inf:", W_inf_torch.item())
    print("NumPy W_inf:", W_inf_numpy)
    print("W_inf Difference:", abs(W_inf_torch.item() - W_inf_numpy))
    print()
    print("PyTorch W_list:")
    print([w.item() for w in W_list_torch])
    print("NumPy W_list:")
    print(W_list_numpy)
    print("W_list Differences:")
    print([
        abs(w_torch.item() - w_numpy)
        for w_torch, w_numpy in zip(W_list_torch, W_list_numpy)
    ])
    print()


def test_gpu_support():
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    # Create random sample and target points
    torch.manual_seed(42)

    # Create sample and target points on CPU
    sample_cpu = torch.rand(5, 2)
    target_cpu = torch.rand(3, 2)
    sample_weight_cpu = torch.ones(5) / 5
    target_weight_cpu = torch.ones(3) / 3

    # Move to GPU
    sample_gpu = sample_cpu.cuda()
    target_gpu = target_cpu.cuda()
    sample_weight_gpu = sample_weight_cpu.cuda()
    target_weight_gpu = target_weight_cpu.cuda()

    # Calculate gradient_EOT on CPU
    zeta = 0.1
    D_nu_obj_cpu, D_y_obj_cpu, cost_cpu = edt.gradient_EOT(sample_cpu,
                                                           sample_weight_cpu,
                                                           target_cpu,
                                                           target_weight_cpu,
                                                           zeta=zeta)

    # Calculate gradient_EOT on GPU
    D_nu_obj_gpu, D_y_obj_gpu, cost_gpu = edt.gradient_EOT(sample_gpu,
                                                           sample_weight_gpu,
                                                           target_gpu,
                                                           target_weight_gpu,
                                                           zeta=zeta)

    # Compare results
    print("GPU Support Test:")
    print("CPU Cost:", cost_cpu.item())
    print("GPU Cost:", cost_gpu.item())
    print("Cost Difference:", abs(cost_cpu.item() - cost_gpu.item()))
    print()
    print("CPU D_nu_obj:")
    print(D_nu_obj_cpu)
    print("GPU D_nu_obj (moved to CPU):")
    print(D_nu_obj_gpu.cpu())
    print("Max D_nu_obj Difference:",
          torch.max(torch.abs(D_nu_obj_cpu - D_nu_obj_gpu.cpu())).item())
    print()


if __name__ == '__main__':
    print("Testing EDOT_disc_torch.py implementation")
    print("=" * 50)

    # Run tests
    test_euclidean_distance()
    test_gradient_EOT()
    test_cont_wasserstein()
    test_gpu_support()

    print("All tests completed!")
