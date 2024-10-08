import sys
sys.stdout.reconfigure(encoding='utf-8')



import tensorflow as tf
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_generator()

            # Recreate optimizer instance
            optimizer = tf.keras.optimizers.Adam()

            # Set the optimizer for the model
            training.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            training.train()
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
