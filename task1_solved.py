import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train(log_every: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2),
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    forward_times = []
    backward_times = []
    cuda_events = []

    loss_sum = torch.zeros((), device=device)
    num_samples = 0

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, (data, target) in enumerate(dataloader):
        # pin_memory + non_blocking сохраняют асинхронную передачу CPU -> GPU.
        data = data.to(device, non_blocking=use_cuda)
        target = target.to(device, non_blocking=use_cuda)

        # Шум создаётся сразу на нужном устройстве, без лишнего CPU -> GPU копирования.
        data.add_(torch.randn_like(data))

        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            start_fwd = torch.cuda.Event(enable_timing=True)
            end_fwd = torch.cuda.Event(enable_timing=True)
            start_bwd = torch.cuda.Event(enable_timing=True)
            end_bwd = torch.cuda.Event(enable_timing=True)

            # CUDA events измеряют реальное время работы GPU без синхронизации на каждом батче.
            start_fwd.record()
            output = model(data)
            loss = criterion(output, target)
            end_fwd.record()

            start_bwd.record()
            loss.backward()
            end_bwd.record()

            cuda_events.append((start_fwd, end_fwd, start_bwd, end_bwd))
        else:
            time_start = time.perf_counter()
            output = model(data)
            loss = criterion(output, target)
            time_end = time.perf_counter()
            forward_times.append(time_end - time_start)

            time_start_bwd = time.perf_counter()
            loss.backward()
            time_end_bwd = time.perf_counter()
            backward_times.append(time_end_bwd - time_start_bwd)

        optimizer.step()

        # Сохраняется только detach-статистика, а не loss с вычислительным графом.
        batch_samples = data.size(0)
        loss_sum += loss.detach() * batch_samples
        num_samples += batch_samples

        if log_every is not None and log_every > 0:
            if batch_idx % log_every == 0 or batch_idx == len(dataloader) - 1:
                # .item() синхронизирует CUDA, поэтому делаем это только при явном логировании.
                print(f"Батч {batch_idx}, loss: {loss.detach().item():.4f}")

    if use_cuda:
        torch.cuda.synchronize(device)
        forward_times = [
            start_fwd.elapsed_time(end_fwd) / 1000
            for start_fwd, end_fwd, _, _ in cuda_events
        ]
        backward_times = [
            start_bwd.elapsed_time(end_bwd) / 1000
            for _, _, start_bwd, end_bwd in cuda_events
        ]

    avg_loss = (loss_sum / num_samples).item()

    print(
        f"Эпоха завершена, средний loss: {avg_loss:.4f}, "
        f"среднее время forward: {statistics.mean(forward_times):.6f} с, "
        f"среднее время backward: {statistics.mean(backward_times):.6f} с"
    )

    if use_cuda:
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Максимально выделенная память: {max_memory_mb:.2f} MiB")


if __name__ == "__main__":
    train()
