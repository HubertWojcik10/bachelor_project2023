import logging 
import psutil

class Logger:
    """
        Custom Logger class
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.logger = None
        self.create_logger()

    def create_logger(self) -> None:
        """
            Create the logger
        """
        logging.basicConfig(filename=self.log_dir, filemode="w",
                format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
                level=logging.DEBUG)
        
        self.logger = logging.getLogger(__name__)

    def log_model_info(self, log_type: str, cur_pearson: float = 0.0, best_pearson: float = 0.0) -> None:
        """
            Log the model info
        """
        if log_type == "start_train":
            self.logger.info("Starting training...")
            print("Starting training...")
        elif log_type == "start_validation":
            self.logger.info("Starting validation...")
            print("Starting validation...")
        elif log_type == "start_prediction":
            self.logger.info("Starting prediction...")
            print("Starting prediction...")
        elif log_type == "finished_prediction":
            self.logger.info(f"Finished prediction with pearson corr: {cur_pearson:.4f}")
            print(f"Finished prediction with pearson corr: {cur_pearson:.4f}")
        elif log_type == "finished_validation":
            self.logger.info(f"Finished validation with pearson corr: {cur_pearson:.4f}, best pearson corr: {best_pearson:.4f}")
            print(f"Finished validation with pearson corr: {cur_pearson:.4f}, best pearson corr: {best_pearson:.4f}")

    def log_epoch_info(self, epoch: int, epochs: int) -> None:
        """ 
            Log the epoch info
        """
        self.logger.info(f"{'-'*25} Epoch {epoch+1} of {epochs} {'-'*25}")
        print(f"{'-'*25} Epoch {epoch+1} of {epochs} {'-'*25}")

    def log_batch_info(self, idx: int, total_batches: int, loss: float) -> None:
        """
            Log the batch info
        """
        self.logger.info(f"batch {idx+1} of {total_batches}")
        self.logger.info(f"loss: {loss:.2f}")

        print(f"batch {idx+1} of {total_batches}")
        print(f"loss: {loss:.2f}")

    def log_memory_info(self) -> None:
        """
            Log the memory info
        """
        self.logger.info(f"Memory used: {psutil.virtual_memory().percent}%")
        print(f"Memory used: {psutil.virtual_memory().percent}%")