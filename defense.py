import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.dimensionality_reduction import calculate_umap_of_gradients
from federated_learning.dimensionality_reduction import calculate_tsne_of_gradients
from federated_learning.parameters import get_layer_parameters
from federated_learning.parameters import calculate_parameter_gradients
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from federated_learning.utils import apply_standard_scaler
from federated_learning.utils import get_worker_num_from_model_file_name
from client import Client
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Paths you need to put in.
MODELS_PATH = "E:\poison\FL\SunWei\DataPoisoning_FL-master-DDPM\TEST//baseline//1002_models"
#EXP_INFO_PATH = "/absolute/path/to/log/file/1823.log"

# The epochs over which you are calculating gradients.
EPOCHS = list(range(1, 201))

# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
LAYER_NAME = "fc.weight"

# The source class.
#CLASS_NUMS=[0,1,2,3,4,5,6,7,8,9]
CLASS_NUMS=[1]
#CLASS_NUMS=[0,3,7,1,9]

# The IDs for the poisoned workers. This needs to be manually filled out.
# You can find this information at the beginning of an experiment's log file.
POISONED_WORKER_IDS = []

# The resulting graph is saved to a file
#SAVE_NAME = "defense_results.jpg"
SAVE_SIZE = (24,24)


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))

        clients.append(client)

    return clients


def plot_gradients_2d(gradients,SAVE_NAME):
    fig = plt.figure()

    for (worker_id, gradient) in gradients:
        if worker_id in POISONED_WORKER_IDS:
            #plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=500, linewidth=5, alpha=0.6)
            plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=500, linewidth=5)
        else:
            #plt.scatter(gradient[0], gradient[1], color="orange", s=180, alpha=0.4)
            plt.scatter(gradient[0], gradient[1], color="orange", s=180)

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.xlim((-20,20))
    plt.ylim((-25,25))
    plt.grid(False)
    plt.margins(0,0)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)

def plot_gradients_3d(gradients,SAVE_NAME):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    for (worker_id, gradient) in gradients:
        if worker_id in POISONED_WORKER_IDS:
            # 绘制中毒工作者的梯度点，使用不同的颜色和标记
            ax.scatter(gradient[0], gradient[1], gradient[2], color="blue", marker="x", s=500, linewidth=5, alpha=0.6)
        else:
            # 绘制正常工作者的梯度点
            ax.scatter(gradient[0], gradient[1], gradient[2], color="orange", s=180, alpha=0.4)

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.xlim((-25,25))
    plt.ylim((-20,20))
    plt.grid(False)#网格线
    plt.margins(0,0,0)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)#紧凑和空白


if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    for CLASS_NUM in CLASS_NUMS:
        param_diff = []
        worker_ids = []

        for epoch in EPOCHS:
            start_model_files = get_model_files_for_epoch(model_files, epoch)
            start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
            start_model_file = os.path.join(MODELS_PATH, start_model_file)
            start_model = load_models(args, [start_model_file])[0]

            start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

            end_model_files = get_model_files_for_epoch(model_files, epoch)
            end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

            for end_model_file in end_model_files:
                worker_id = get_worker_num_from_model_file_name(end_model_file)
                end_model_file = os.path.join(MODELS_PATH, end_model_file)
                end_model = load_models(args, [end_model_file])[0]

                end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

                gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
                gradient = gradient.flatten()

                param_diff.append(gradient)
                worker_ids.append(worker_id)

        logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))

        logger.info("Prescaled gradients: {}".format(str(param_diff)))
        scaled_param_diff = apply_standard_scaler(param_diff)
        logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
        dim_reduced_gradients = calculate_pca_of_gradients(logger, scaled_param_diff, 3)
        #dim_reduced_gradients = calculate_tsne_of_gradients(logger, scaled_param_diff, 2)
        #dim_reduced_gradients = calculate_umap_of_gradients(logger, scaled_param_diff, 2)
        logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))

        logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))


        SAVE_NAME = "results_{}.jpg".format(CLASS_NUM)
        plot_gradients_3d(zip(worker_ids, dim_reduced_gradients),SAVE_NAME)

        #plot_gradients_2d(zip(worker_ids, dim_reduced_gradients))
