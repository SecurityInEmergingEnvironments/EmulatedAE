MODEL_SAVE_PARENT_PATH = "data"
CIFAR100_LABEL_TYPE = 'coarse_label'
CIFAR10_LABEL_TYPE = 'label'
TMP_NAME = "/tmp_"
GTSRB_RAW_DATA_PATH = "data_224_224/datasets/GTSRB_raw_data/"
MIN_VAL_LOSS_CONSTANT = 9999999
CIFAR100_ORIGINAL_SIZE = 32
VAE_DEFAULT_INPUT_SHAPE = 32
ALL_NN_DENOISER_NAMES = ['ae', 'vae', 'unet', 'Chained_Vae_EAE', 'Chained_EAE_Vae']
ALL_REGULAR_DENOISER_NAMES = ['emulated-ae', 'jpegCompression', 'non-local', 'tv', 'noise2self', 'eae_jpeg', 'jpeg_eae']
BATCH_SIZE = 20

AE_REGULAR_TRAINING_EPOCHS = 5
VAE_REGULAR_TRAINING_EPOCHS = 200
UNET_REGULAR_TRAINING_EPOCHS = 1

DEFENDER_NAME = "defenders"
BASELINE_PERFORMANCE = "baseline_performance"
DEFENDER_PERFORMANCE = 'defender_performance'
ATTACKER_PERFORMANCE = "attacker_performance"
ATTACKER_PERFORMANCE_METRICS = ['robust_accuracy']
BENIGN_DEFENDER_PERFORMANCE_METRICS = ['natural_accuracy']
ROBUST_DEFENDER_PERFORMANCE_METRICS = ['robust_accuracy']

CSV_HEADER = [
  "benign_acc",       "PGD_adv_acc",        "E-AE_denoiser_for_PGD", "DeepFool_adv_acc",        "E-AE_denoiser_for_DF",
  "benign_precision", "PGD_adv_precision",  "E-AE_denoiser_for_PGD", "DeepFool_adv_precision",  "E-AE_denoiser_for_DF",
  "benign_recall",    "PGD_adv_recall",     "E-AE_denoiser_for_PGD", "DeepFool_adv_recall",     "E-AE_denoiser_for_DF",
  "benign_fscore",    "PGD_adv_f1_score",   "E-AE_denoiser_for_PGD", "DeepFool_adv_f1_score",   "E-AE_denoiser_for_DF",
  ]
