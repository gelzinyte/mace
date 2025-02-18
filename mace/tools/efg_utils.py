import numpy as np
import logging
import ast

from mace import data, modules, tools


def get_efgs_per_element_weights(loss_element_weights, train_loader, z_table):
        
    if loss_element_weights is None:
        efgs_element_weights = 1/modules.scaling_classes["efgs_mean_eval_scaling"](data_loader=train_loader)
        logging.info(f"No efg per-element loss weights were given, computing from data")
    else:
        logging.info(f"Using efg per-element loss weights from command line") 
        try:
            efgs_element_weights_dict = ast.literal_eval(loss_element_weights)
        except Exception as e:
            raise RuntimeError(f"Loss element weights specified invalidly, error {e} occured") from e

        efgs_element_weights = np.array([efgs_element_weights_dict[z] for z in z_table.zs])

    logging.info(f'using weighted efg loss with weights: {efgs_element_weights}')

    return efgs_element_weights

