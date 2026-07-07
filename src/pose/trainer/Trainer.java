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
			while (loader.hasNext())
			{
				Batch batch = loader.next();

				Tensor prediction = layer.forward(batch.getImages());
				Tensor target = batch.getHeatmaps();

				for (int i = 10000; i < 10020; i++)
				{
					System.out.printf("%d  pred=%f  target=%f%n", i, prediction.getData()[i], target.getData()[i]);
				}
				float loss = lossFunction.forward(prediction, target);
				Tensor gradient = lossFunction.backward(prediction, target);

				layer.backward(gradient);

				optimizer.step(layer);

				Tensor.release(batch.getImages());
				Tensor.release(batch.getHeatmaps());
				Tensor.release(prediction);
				Tensor.release(target);
				Tensor.release(gradient);

				System.out.println("Epoch " + (epoch + 1) + " | Loss: " + loss);
			}
			loader.reset();
			loader.shuffle();
			System.out.println("Epoch: " + (epoch + 1));
		}
	}
}
