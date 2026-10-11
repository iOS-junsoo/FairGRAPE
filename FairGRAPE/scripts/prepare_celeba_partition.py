"""Fetch only CelebA's official partition file and verify the upstream checksum."""
import argparse
from pathlib import Path
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from celeba_single import PARTITION_FILE_ID, read_partition


def prepare(output):
    output = Path(output)
    if output.exists():
        read_partition(output)
        print(f'Official partition verified: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / 'list_eval_partition.txt'
        url = f'https://drive.google.com/uc?export=download&id={PARTITION_FILE_ID}'
        with urllib.request.urlopen(url, timeout=60) as response:
            candidate.write_bytes(response.read())
        read_partition(candidate)
        # Never overwrite a concurrently supplied file.
        with output.open('xb') as target:
            target.write(candidate.read_bytes())
    print(f'Official partition downloaded and verified: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='csv/list_eval_partition.txt')
    prepare(parser.parse_args().output)
