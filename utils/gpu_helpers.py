def deallocate_gpu_memory_if_low(model:object=None) -> None:
    """Delete model if low on GPU resources"""
    import torch

    if torch.cuda.is_available():
        device = torch.cuda.current_device()

        # Get the total and used memory in bytes
        total_mem = torch.cuda.get_device_properties(device).total_memory
        used_mem = torch.cuda.memory_allocated(device)

        # Convert the memory size to GB
        total_mem_gb = total_mem / (1024 * 1024 * 1024)
        used_mem_gb = used_mem / (1024 * 1024 * 1024)

        # Calculate remaining memory
        remaining_mem_gb = total_mem_gb - used_mem_gb
        remaining_mem_percent = remaining_mem_gb / total_mem_gb * 100
        print(
            f"GPU memory: Total {total_mem_gb:.2f}GB, Used {used_mem_gb:.2f}GB, Remaining {remaining_mem_gb:.2f}GB ({remaining_mem_percent:.2f}%)",
        )

        if remaining_mem_percent <= 50:
            print(
                f"Clearing up GPU memory. {used_mem_gb:.1f}GB of {total_mem_gb:.2f}GB used, {remaining_mem_percent:.1f}% remaining.",
            )
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            if model:
                del model


def check_gpu_availability() -> bool:
    import torch

    return torch.cuda.is_available()