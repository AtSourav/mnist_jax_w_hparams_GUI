"""Modifying the tensorboard summary writer so as to include graphs under the hparams tab."""

import os
import time
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import hparams

class SummaryWriter_mod(SummaryWriter):
    """
    Modifying the tensorboard summary writer so as to include graphs under the hparams tab by subclassing the SummaryWriter class and adding a new function that is a modification of add_hparams.
    """

    def __init__(
            
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    
    ):
        
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)



    def add_hparams_plot(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None, global_step=None):

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            run_name = str(time.time())
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp, global_step)
            w_hp.file_writer.add_summary(ssi, global_step)
            w_hp.file_writer.add_summary(sei, global_step)
            for k, v in metric_dict.items():                # v should be a list
                step = 1
                for value in v:
                    w_hp.add_scalar(k, value, step)
                    step += 1

               
 

		