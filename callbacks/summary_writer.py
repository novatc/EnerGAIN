from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:

            data = self.locals["infos"][0]
            for key in data:
                self.tb_formatter.writer.add_scalar("train/" + key, data[key], self.num_timesteps)

            self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
            self.tb_formatter.writer.flush()
