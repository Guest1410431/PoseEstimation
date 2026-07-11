package pose.trainer;

import pose.layer.Layer;
import pose.tensor.Tensor;

public class Trainer
{
	private Layer layer;
	private Dataset dataset;
	private Optimizer optimizer;
	private LossFunction lossFunction;

	public Trainer(Layer layer, Dataset dataset, Optimizer optimizer, LossFunction lossFunction)
	{
		this.layer = layer;
		this.dataset = dataset;
		this.optimizer = optimizer;
		this.lossFunction = lossFunction;
	}

	public void train(int epochs)
	{
		DataLoader loader = new DataLoader(dataset, 16);

		for (int epoch = 0; epoch < epochs; epoch++)
		{
			float epochLoss = 0;
			int batches = 0;
			
			while (loader.hasNext())
			{
				Batch batch = loader.next();

				Tensor prediction = layer.forward(batch.getImages());
				Tensor target = batch.getHeatmaps();
				
				//System.out.println("Prediction min: " + prediction.min() + " | max: " + prediction.max() + " | mean: " + prediction.mean());
				//System.out.println("Target---- min: " + target.min() + " | max: " + target.max() + " | mean: " + target.mean());
				
				float loss = lossFunction.forward(prediction, target);
				epochLoss += loss;
				batches++;
				
				Tensor gradient = lossFunction.backward(prediction, target);
				layer.backward(gradient);

				optimizer.step(layer);

				Tensor.release(batch.getImages());
				Tensor.release(prediction);
				Tensor.release(target);
				Tensor.release(gradient);
			}
			System.out.println("Epoch " + (epoch + 1) + " | Loss: " + epochLoss / batches);
			loader.reset();
			loader.shuffle();
		}
	}
}
