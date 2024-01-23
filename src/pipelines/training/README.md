## Training image
If you have a local registry you can use `localhost:5000` as your `<your_acc>`
### Building and pushing your image

Begin in the `src` directory because Dockerfile will build with this as the parent. This is important because the training pipeline requires `src/components/` directory.
```
cd src/
```

```
docker build -t <your_acc>/mnist-training:latest -f pipelines/training/Dockerfile .
```

You can push the image via:
```
docker push <your_acc>/mnist-training:latest
```

Clean up the local version of the registry:
```
docker rm <your_acc>/mnist-training:latest
```

### Using your training image

Pull the image from your registry
```
docker pull <your_acc>/mnist-training:latest
```
Run the latest training image
```
docker run <your_acc>/mnist-training python3 opt/training-pipeline.py \
    --epochs <epoch number>
    --l1 <layer1_regularization>
    --l2 <layer2_regularization>
    --num_hidden <l1 units>
    --learning_rate <learning_rate>
```