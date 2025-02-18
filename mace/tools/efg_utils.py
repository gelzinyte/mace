import numpy as np
import logging
import ast

from mace import data, modules, tools


def get_efgs_per_element_weights(loss_element_weights_multiplier, train_loader, z_table):
        
    efgs_element_weights = 1/modules.scaling_classes["efgs_mean_eval_scaling"](data_loader=train_loader)
    logging.info(f"Computed per-element loss weights from data: {efgs_element_weights}")
    if loss_element_weights_multiplier is not None:
        try:
            efgs_element_weights_multiplier_dict = ast.literal_eval(loss_element_weights_multiplier)
        except Exception as e:
            raise RuntimeError(f"Loss element weights specified invalidly, error {e} occured") from e

        logging.info(f"Read a multiplier for per-element loss weights from command line: {efgs_element_weights_multiplier_dict}") 
        efgs_element_weights_multiplier = np.array([efgs_element_weights_multiplier_dict[z] for z in z_table.zs])

        efgs_element_weights *=  efgs_element_weights_multiplier

    logging.info(f'Using weighted efg loss with weights: {efgs_element_weights}')

    return efgs_element_weights

