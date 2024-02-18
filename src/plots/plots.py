import matplotlib.pyplot as plt
from datetime import datetime

class Plots:
    @staticmethod
    def plot_loss(losses):
        num_epochs = len(losses)
        fig, ax = plt.subplots(1, num_epochs, figsize=(20, 8))
        for i, (epoch, loss) in enumerate(losses.items()):
            ax[i].plot(loss, label=f"epoch {epoch+1}")
            ax[i].set_title(f"epoch {epoch+1}")
            ax[i].set_xlabel("batch")
            ax[i].set_ylabel("loss")

        plt.show()
        curr_time = datetime.now()
        fig.savefig(f"plots/loss_{curr_time}.png")
