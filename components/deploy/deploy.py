import mlflow
import io
import base64
import tensorflow as tf
from PIL import Image
from typing import Dict
from kserve import Model, ModelServer


# project imports
from components.preprocess_step import preprocess

class MNISTModel(Model):
    def __init__(self, name: str): 
        super().__init__(name)
    
    def load(self):
        # load model from mlflow
        try: 
            results = mlflow.search_registered_models(
                filter_string='name = "mnist-hyperparam-local"')
            latest_model_details = results[0].latest_versions[0]
            self.model = mlflow.tensorflow.load_model(
                model_uri=f'{latest_model_details.source}')
            print(f'Successfully loaded model from {latest_model_details.source}')
        except IndexError:
            print('No models found. Cannot perform inference.')
            return None

    def predict(self, payload: Dict):
        # file check and load
        # extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        # if not extension:
        #     return "Image must be jpg or png format."
        # await file.read()
        # image = np.array(Image.open(file.file)) 
        # process image
        img_data = payload["instances"][0]["image"]["b64"]
        raw_img_data = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(raw_img_data))

        # preprocess image and predict
        image, _ = preprocess.preprocess_mnist_tfds(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        result = self.model.predict(image).argmax()
        return {"predictions": result}
    

if __name__ == "__main__":
    model = MNISTModel("mnist on kubeflow")
    ModelServer().start([model])