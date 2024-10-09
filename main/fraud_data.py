from torch.utils.data import Dataset

from custom_data_loader import CustomDataLoader
from logger import Logger

logger: Logger = Logger().get_logger()


class FraudDataset(Dataset):

    def __init__(self, data_loader: CustomDataLoader, file_path: str, file_name: str, transform_to_tensor=None,
                 target_transform=None):
        self.data_loader = data_loader
        self.file_path = file_path
        self.file_name = file_name
        self.transform_to_tensor = transform_to_tensor
        self.target_transform = target_transform

    # TODO: Limit the amount of times we go to the data set
    def __len__(self) -> int:
        num_rows: int = len(
            self.data_loader.get_data_frame_from_zip_file(file_path=self.file_path, file_name=self.file_name))
        return num_rows

    # TODO: Add return datatype
    def __getitem__(self, idx):
        pass
