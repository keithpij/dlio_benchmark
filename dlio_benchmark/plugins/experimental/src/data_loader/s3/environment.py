import argparse
import platform

import psutil

def get_memory_usage() -> dict:
    '''
    Return memory usage information for the current system.

    Returns a dict with keys:
      total, available, used, free, percent, cached, buffers, swap_total, swap_used
    Sizes are in bytes, percent is a float (0-100).
    '''

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "total": int(vm.total),
        "available": int(getattr(vm, "available", vm.free)),
        "used": int(vm.used),
        "free": int(vm.free),
        "percent": float(vm.percent),
        "cached": int(getattr(vm, "cached", 0)),
        "buffers": int(getattr(vm, "buffers", 0)),
        "swap_total": int(sw.total),
        "swap_used": int(sw.used),
    }


def show_drives_space():
    """
    Displays the available space on each mounted drive across all operating systems.
    Requires the 'psutil' library: pip install psutil
    On macOS, shows only the root filesystem and volumes under /Volumes to avoid duplicates.
    """
    partitions = psutil.disk_partitions()
    system = platform.system()
    filtered_partitions = []
    
    if system == 'Darwin':  # macOS
        for partition in partitions:
            if partition.mountpoint == '/' or partition.mountpoint.startswith('/Volumes/'):
                filtered_partitions.append(partition)
    else:
        filtered_partitions = partitions
    
    for partition in filtered_partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            print(f"Drive: {partition.device} mounted at {partition.mountpoint}")
            print(f"Total: {usage.total / (1024**3):.2f} GB")
            print(f"Used: {usage.used / (1024**3):.2f} GB")
            print(f"Free: {usage.free / (1024**3):.2f} GB")
            print(usage)
        except PermissionError:
            print(f"Could not access {partition.mountpoint}")


def main():
    '''
    Main function that processes the arguments sent to this module via the command line.
    '''

    # Setup the command line options.
    parser = argparse.ArgumentParser(description='Simulation Command line interface.')
    parser.add_argument('-mu', '--memory_usage', help='Memory usage.', action='store_true')
    parser.add_argument('-du', '--disk_usage', help='Disk usage.', action='store_true')

    args = parser.parse_args()

    if args.memory_usage:
        memory_usage = get_memory_usage()
        print(f'Total: {memory_usage["total"]} bytes')
        print(f'Available: {memory_usage["available"]} bytes')
        print(f'Used: {memory_usage["used"]} bytes')
        print(f'Free: {memory_usage["free"]} bytes')
        print(f'Percent: {memory_usage["percent"]}%')
        print(f'Cached: {memory_usage["cached"]} bytes')
        print(f'Buffers: {memory_usage["buffers"]} bytes')
        print(f'Swap Total: {memory_usage["swap_total"]} bytes')
        print(f'Swap Used: {memory_usage["swap_used"]} bytes')  

    if args.disk_usage:
        show_drives_space()


if __name__ == '__main__':
    main()
