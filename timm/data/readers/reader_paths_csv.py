"""
A dataset reader that extracts images from a single folder
based on a csv with labels and filenames relative to that folder.
"""
import os
import pandas as pd

from .reader import Reader


class ReaderPathsCsv(Reader):
    def __init__(
            self,
            images_dir,
            samples_csv_path,
            class_map: dict[str, int],
    ):
        super().__init__()
        assert isinstance(class_map, dict)

        self.images_dir = images_dir
        samples_df = pd.read_csv(samples_csv_path).astype(str)

        if not samples_df["label"].isin(class_map).all():
            unrecognized_ids = ~samples_df["label"].isin(class_map)
            unrecognized_labels = set(samples_df.loc[unrecognized_ids, "label"])
            raise ValueError(f"Unrecognized labels found in samples_df: {unrecognized_labels}")
        
        samples_df["label"] = samples_df["label"].map(class_map)
        
        self.samples_df = samples_df

    def __getitem__(self, index):
        filename, target = self.samples_df.iloc[index]
        path = os.path.join(self.images_dir, filename)
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples_df.index)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples_df.iloc[index, "filename"]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.images_dir)
        return filename
