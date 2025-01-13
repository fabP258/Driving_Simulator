import os
from pathlib import Path
from trainer import VqVaeTrainer


def main():

    output_path = "/home/fabio/Repos/DrivingPolicyTrainer/output"

    segment_folders = []

    for chunk_idx in range(4, 7):
        chunk_path = "/home/fabio/comma2k19/Chunk_" + f"{chunk_idx}"
        print(chunk_path)
        baselevel = len(chunk_path.split(os.path.sep))
        for subdirs, dirs, files in os.walk(chunk_path):
            curlevel = len(subdirs.split(os.path.sep))
            if (curlevel - baselevel) == 2:
                segment_folders.append(Path(subdirs))
    output_path = Path(__file__).parent / "output"
    output_path.mkdir(exist_ok=True, parents=True)

    trainer = VqVaeTrainer(segment_folders, output_path, batch_size=5, num_workers=4)
    trainer.train(25)


if __name__ == "__main__":
    main()
