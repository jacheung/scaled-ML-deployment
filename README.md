# Scaled ML Deployment
This repository is part 3, the last part, of a series of templates for stepping up a POC deployment to one that is scalable in production. This part is built on top of Kubernetes and introduces:
 1. Kubeflow pipelines
 2. Seldon/KServe 

 ![](/docs/vision-scaled.png) 



 ### Training image
If you have a local registry you can use `localhost:5000` as your `<your_acc>`
```
docker build -t <your_acc>/mnist-training:latest .
```



 ```
    "image": "docker.io/misohu/kubeflow-training:latest",
    "command": [
        "python",
        "/opt/training-pipeline.py",
        "--s3-storage=true",
        f"{tf_model}",
        "--mlflow-model-name=ml-workflow-demo-model",
        f"--bucket={bucket}",
        f"--bucket-key={key}",
    ]
```