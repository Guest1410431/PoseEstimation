package pose.trainer;

import pose.layer.Layer;
import pose.layer.Tensor;

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
		for (int epoch = 0; epoch < epochs; epoch++)
		{
			dataset.shuffle();
			
			float epochLoss = 0f;
			
			for(int i=0; i<dataset.size(); i++)
			{
				TrainingSet sample = dataset.get(i);
				
				Tensor prediction = layer.forward(sample.getImage());
				
				float loss = lossFunction.forward(prediction, sample.getTargetHeatmaps());
				epochLoss += loss;
				
				Tensor gradient = lossFunction.backward(prediction, sample.getTargetHeatmaps());
				
				layer.backward(gradient);
				optimizer.step(layer);
			}
			System.out.println("Epoch " + epoch + " Loss: " + epochLoss / dataset.size());
		}
	}
}














