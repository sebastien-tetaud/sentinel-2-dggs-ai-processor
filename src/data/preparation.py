

def prepare_data(config):
    """
    Prepare the data loader for test datasets.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        DataLoader: Data loader for the test dataset.
    """
    version = config['DATASET']['version']
    resize = config['TRAINING']['resize']

    TEST_DIR = f"/mnt/disk/dataset/sentinel-ai-processor/{version}/test/"
    df_test_input, df_test_output = prepare_paths(TEST_DIR)

    test_dataset = Sentinel2Dataset(df_x=df_test_input, df_y=df_test_output,
                                     train=True, augmentation=False, img_size=resize)
    logger.info(df_test_output.head(5))

    return define_loaders(
        train_dataset=test_dataset,
        val_dataset=None,
        train=False,
        batch_size=config['TRAINING']['batch_size'],
        num_workers=config['TRAINING']['num_workers']
    )